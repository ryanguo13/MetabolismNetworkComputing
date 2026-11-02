% Run with Octave from the repository root. Original publisher sources are
% extracted to ignored results/scratch/k-ecoli457 by the Python audit script.
% Preserve source files; optimize only the identical mass-action products.
function kecoli_reference(root)
  addpath(fullfile(root, 'results/scratch/k-ecoli457'));
  input = load(fullfile(root, 'results/scratch/kecoli-input.mat'));
  load(fullfile(root, 'results/scratch/k-ecoli457/Data.mat'));
  % Original perturbModel and parallel_automated implement the chosen case.
  [x1, x2] = parallel_automated(1, 2, aerobic_enzyme_fraction);
  y0 = perturbModel(network_data, metabolite_and_enzyme_conc, x1, x2);
  initial_difference = max(abs(y0(:) - input.y0(:)));
  S = sparse(network_data.S_f_b);
  [ri, rj, orders] = find(-min(S, 0));
  k = elementary_kinetic_parameter(:);
  rhs = @(t,y) S * (k .* accumarray(rj, y(ri).^orders, [length(k), 1], @prod, 1));
  reference_rhs = zeros(size(input.check_states));
  optimized_rhs = zeros(size(input.check_states));
  jacobian = sparse(length(y0), length(y0));
  for i = 1:columns(input.check_states)
    state = input.check_states(:,i);
    reference_rhs(:,i) = mass_balance_ode(0, state, k, full(S));
    optimized_rhs(:,i) = rhs(0, state);
  end
  jacobian = sparse(jacobFun_Big_447_Model(0, input.check_states(:,end), k));
  options = odeset('RelTol', 1e-10, 'AbsTol', 1e-12, ...
                   'Jacobian', @(t,y) sparse(jacobFun_Big_447_Model(t,y,k)));
  started = tic;
  [times, concentrations] = ode15s(rhs, input.times(:), y0, options);
  integration_seconds = toc(started);
  octave_version = version;
  % Fixed output window intentionally replaces the source steady-state/CPU
  % stop event; this is registered explicitly for all benchmark solvers.
  save('-mat7-binary', fullfile(root, 'results/m1-completion/kecoli-octave.mat'), ...
       'times', 'concentrations', 'reference_rhs', 'optimized_rhs', ...
       'jacobian', 'initial_difference', 'integration_seconds', 'octave_version');
  fprintf('Octave source audit and ode15s trajectory saved: %.6f seconds\n', integration_seconds);
end
