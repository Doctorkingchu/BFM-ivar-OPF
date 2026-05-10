% ACOPF.m
% ------------------------------------------------------------------------
% AC-OPF baseline on the IEEE 3012-bus (Polish winter 2007-08 evening peak)
% case, intended as the MATLAB/MATPOWER reference (MATPOWER MIPS) against
% which the Python BFM_ivar_mit driver is compared.
%
% Prerequisite
% ------------
%   MATPOWER (>= 7.0) on the MATLAB path.
%   If not yet installed, see https://matpower.org/download and run
%       install_matpower
%   once in MATLAB. `case3012wp.m` in this folder shadows MATPOWER's
%   bundled copy so local edits win over the library copy.
%
% Cost model
% ----------
%   The case ships with polynomial gencost, NCOST = 3 (cubic-sized row),
%   but the shipped c2 coefficients are zero on every generator, making
%   the cost effectively linear. To exercise a genuinely quadratic cost
%   (as the BFMag comparison asks for), set INJECT_QUADRATIC_C2 = true
%   below; a small convex-quadratic term is synthesized from each
%   generator's linear c1 so the ordering of generators is preserved.
%
% Shunt / OLTC handling
% ---------------------
%   MATPOWER's standard ACOPF does not treat shunt Gs/Bs or transformer
%   taps as decision variables - they are already fixed parameters, so
%   running with DISABLE_SHUNTS_AND_OLTC = false is the "no controllable
%   shunt, no controllable OLTC" baseline and is the most direct
%   apples-to-apples comparison against BFMag.
%   Set DISABLE_SHUNTS_AND_OLTC = true to go further and zero out the
%   bus shunt admittances plus force every transformer tap to 1.0
%   (MATPOWER encoding: branch(:,9) = 0). This is more aggressive and
%   can stress feasibility on multi-voltage grids.
% ------------------------------------------------------------------------

clear; clc;

%% ---- Options ----------------------------------------------------------
DISABLE_SHUNTS_AND_OLTC = false;  % keep case3012wp's fixed shunts/taps (zeroing Gs/Bs collapses ACOPF voltage profile -- 10 GVAr of reactive load needs capacitor support)
INJECT_QUADRATIC_C2     = false;  % keep shipped linear cost (c2=0) for fair vs BFMivar comparison
C2_SCALE                = 1.0e-3; % c2_new = C2_SCALE * c1 (per MW^2)
RESULT_FILE             = 'ACOPF_results.txt';

%% ---- Start result log -------------------------------------------------
if exist(RESULT_FILE, 'file'); delete(RESULT_FILE); end
diary(RESULT_FILE);
diary on;

fprintf('[INFO] Run timestamp: %s\n', datestr(now, 'yyyymmdd_HHMMSS'));

%% ---- Load case --------------------------------------------------------
fprintf('[INFO] Loading case3012wp (Polish winter 2007-08 evening peak)\n');
mpc = case3012wp;
fprintf('[INFO] buses = %d,  branches = %d,  gens = %d\n', ...
        size(mpc.bus,1), size(mpc.branch,1), size(mpc.gen,1));
fprintf('[INFO] gencost MODEL = %d (2 = polynomial), NCOST = %d\n', ...
        mpc.gencost(1,1), mpc.gencost(1,4));

%% ---- Optionally disable shunts and OLTC taps --------------------------
if DISABLE_SHUNTS_AND_OLTC
    fprintf('[INFO] Zeroing bus Gs/Bs and forcing branch tap = 1.0\n');
    mpc.bus(:,5)     = 0;   % Gs
    mpc.bus(:,6)     = 0;   % Bs
    mpc.branch(:,9)  = 0;   % tap ratio (0 -> MATPOWER treats as 1.0)
    mpc.branch(:,10) = 0;   % phase shift angle (deg)
else
    fprintf('[INFO] Keeping case3012wp shunts and taps at their fixed values\n');
end

%% ---- Optionally convert linear cost to genuine quadratic --------------
if INJECT_QUADRATIC_C2
    fprintf(['[INFO] Synthesizing quadratic cost from shipped linear cost\n' ...
             '       c2_new = %.3e * c1     (gencost col 5)\n'], C2_SCALE);
    mpc.gencost(:,5) = C2_SCALE * mpc.gencost(:,6);
else
    fprintf('[INFO] Using shipped linear cost (c2 = 0 on every generator)\n');
end

%% ---- Solver options ---------------------------------------------------
mpopt = mpoption('opf.ac.solver', 'MIPS', ...
                 'mips.max_it',   200,    ...
                 'mips.feastol',  1e-6,   ...
                 'mips.gradtol',  1e-6,   ...
                 'mips.costtol',  1e-6,   ...
                 'verbose',       2,      ...
                 'out.all',       1);

%% ---- Solve ------------------------------------------------------------
fprintf('[INFO] Starting AC-OPF solve with MIPS\n');
t0 = tic;
results = runopf(mpc, mpopt);
wall_time = toc(t0);

%% ---- Summary ----------------------------------------------------------
fprintf('\n===================================================\n');
fprintf(' ACOPF baseline summary (case3012wp)\n');
fprintf('---------------------------------------------------\n');
fprintf('   DISABLE_SHUNTS_AND_OLTC = %d\n', DISABLE_SHUNTS_AND_OLTC);
fprintf('   INJECT_QUADRATIC_C2     = %d  (C2_SCALE = %.3e)\n', ...
        INJECT_QUADRATIC_C2, C2_SCALE);
fprintf('   wall time = %.2f s\n',  wall_time);
fprintf('   success   = %d\n',      results.success);

if results.success
    total_pg   = sum(results.gen(:,2));
    total_qg   = sum(results.gen(:,3));
    total_pd   = sum(results.bus(:,3));
    total_qd   = sum(results.bus(:,4));
    active_loss = total_pg - total_pd;
    vm         = results.bus(:,8);

    fprintf('   objective = %.4f ($/hr)\n', results.f);
    fprintf('   Total Pg  = %10.2f MW\n',   total_pg);
    fprintf('   Total Qg  = %10.2f MVAr\n', total_qg);
    fprintf('   Total Pd  = %10.2f MW\n',   total_pd);
    fprintf('   Total Qd  = %10.2f MVAr\n', total_qd);
    fprintf('   Active loss (Pg - Pd) = %.2f MW (%.3f %% of Pd)\n', ...
            active_loss, 100 * active_loss / total_pd);
    fprintf('   Vm min / max = %.4f / %.4f p.u.\n', min(vm), max(vm));
else
    fprintf('   [WARN] OPF did not converge. Check solver log above.\n');
end
fprintf('===================================================\n');

%% ---- Save .mat snapshot for downstream comparison ---------------------
if results.success
    snapshot = struct();
    snapshot.case_name              = 'case3012wp';
    snapshot.DISABLE_SHUNTS_AND_OLTC = DISABLE_SHUNTS_AND_OLTC;
    snapshot.INJECT_QUADRATIC_C2    = INJECT_QUADRATIC_C2;
    snapshot.C2_SCALE               = C2_SCALE;
    snapshot.wall_time              = wall_time;
    snapshot.objective              = results.f;
    snapshot.gen                    = results.gen;
    snapshot.bus                    = results.bus;
    snapshot.branch                 = results.branch;
    snapshot.gencost                = results.gencost;
    save('ACOPF_snapshot.mat', '-struct', 'snapshot');
    fprintf('[INFO] Snapshot saved to ACOPF_snapshot.mat\n');
end

diary off;
fprintf('[INFO] Log file: %s\n', RESULT_FILE);
