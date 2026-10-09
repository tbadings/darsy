function [] = plot_sol(Nx,Ny, P_reach_lower)
% Quick 2D plot of the reachability probability

% State-space boundaries (replace with your actual values)
x_min = -5;
x_max = 5;

y_min = -5;
y_max = 5;

% Remove the absorbing state (last entry)
V = P_reach_lower(1:end-1);

% Reshape into the 2D partition
% Python uses row-major ordering; MATLAB uses column-major ordering
V_grid = reshape(V, [Ny, Nx])';

% Cell centers
x = linspace(x_min, x_max, Nx+1);
y = linspace(y_min, y_max, Ny+1);

x_centers = (x(1:end-1) + x(2:end)) / 2;
y_centers = (y(1:end-1) + y(2:end)) / 2;

% Plot
figure;

imagesc(x_centers, y_centers, V_grid');

set(gca, 'YDir', 'normal');

colormap(parula);
colorbar;
% clim([0 1]);

xlabel('Position');
ylabel('Velocity');
title('Lower Reachability Probability');

axis tight;
end