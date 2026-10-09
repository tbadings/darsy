function [P_reach_lower, P_reach_upper, sigma] = max_reach_probability(user_param, initial_condition, solution, solver_opts)
% Solve the maximal reachability probability problem
% Currently it only uses lower bound to compute the optimal strategy

max_iter = user_param.max_iter;
tol_VI = solver_opts.tol_VI;
Na = user_param.Na;

% Initialize interval value
V_lower = initial_condition.V_lower;
V_upper = initial_condition.V_upper;
sigma = initial_condition.sigma;
V1_lower = zeros(user_param.Ns+1,1);
V1_upper = zeros(user_param.Ns+1,1);
V1_lower_s_a = zeros(Na,1);

% If state q \in Q_target
for s = user_param.S_target
    V_lower(s) = 1;
    V1_lower(s) = 1;
    V_upper(s) = 1;
    V1_upper(s) = 1;
end


% Interval value iteration

% Iterate the lower bound
tic
for i = 1:max_iter
    % Non-terminal states
    for s = user_param.S_not_terminal
        for a = 1:Na
            P_M = solution.P_lower{s,a}; % Assign lower bound to all
            mysum = sum(P_M); % Get sum of lower bounds for fixed q and a
            % Order interval values
            V_lower_proj_Sp = V_lower(solution.Sp{s,a});
            [~, order_V_lower] = sortrows(V_lower_proj_Sp, 'ascend');
            for spi = 1:length(V_lower_proj_Sp)
                spi_ordered = order_V_lower(spi);
                if (mysum + solution.P_upper{s,a}(spi_ordered) - solution.P_lower{s,a}(spi_ordered) < 1)
                    mysum = mysum + solution.P_upper{s,a}(spi_ordered) - solution.P_lower{s,a}(spi_ordered);
                    P_M(spi_ordered) = solution.P_upper{s,a}(spi_ordered);
                else
                    P_M(spi_ordered) = P_M(spi_ordered) + 1 - mysum;
                    break;
                end
            end
            V1_lower_s_a(a) = P_M'*V_lower_proj_Sp;
        end
        % Maximize in a \in Act
        %             V_lower(q) = max(V_lower_q_a);
        [V1_lower_max, a_max] = max(V1_lower_s_a);
        V1_lower_max = min(1, V1_lower_max);
        if ( V1_lower_max > V_lower(s) )
            V1_lower(s) = V1_lower_max;
            sigma(s) = a_max;
        else
            V1_lower(s) = V_lower(s);
        end
    end

    err = max(abs(V1_lower - V_lower));
    V_lower = V1_lower;

    toc
    fprintf('Iteration = %d . \n', i)
    fprintf('Diference between consecutive value functions = %f . \n', err)

    if ( (i == max_iter) || (err < tol_VI) )
        P_reach_lower = V_lower;
        break;
    end
    % Just for debugging purposes
    %if (i == 1)
    %    estimated_time = user_param.max_iter*toc/60;
    %    fprintf('Estimated value iteration time = %d min. \n', estimated_time)
    %end

end
% Or don't iterate
% P_reach_lower = V_lower;

% Iterate the upper bound
for i = 1:max_iter
    % Non-terminal states
    for s = user_param.S_not_terminal
        a = sigma(s);
        P_M = solution.P_lower{s,a}; % Assign lower bound to all
        mysum = sum(P_M); % Get sum of lower bounds for fixed q and a
        % Order interval values
        V_upper_proj_Sp = V_upper(solution.Sp{s,a});
        [~, order_V_upper] = sortrows(V_upper_proj_Sp, 'descend');
        for spi = 1:length(V_upper_proj_Sp)
            spi_ordered = order_V_upper(spi);
            if (mysum + solution.P_upper{s,a}(spi_ordered) - solution.P_lower{s,a}(spi_ordered) < 1)
                mysum = mysum + solution.P_upper{s,a}(spi_ordered) - solution.P_lower{s,a}(spi_ordered);
                P_M(spi_ordered) = solution.P_upper{s,a}(spi_ordered);
            else
                P_M(spi_ordered) = P_M(spi_ordered) + 1 - mysum;
                break;
            end
        end
        V1_upper_max = P_M'*V_upper_proj_Sp;
        V1_upper_max = min(V1_upper_max, 1);
        if (V1_upper_max > V_upper(s))
            V1_upper(s) = V1_upper_max;
        else
            V1_upper(s) = V_upper(s);
        end
    end

    err = max(abs(V1_upper - V_upper));
    V_upper = V1_upper;

    toc
    fprintf('Iteration = %d . \n', i)
    fprintf('Diference between consecutive value functions = %f . \n', err)

    if ( (i == max_iter) || (err < tol_VI) )
        P_reach_upper = V_upper;
        break;
    end
    % Just for debugging purposes
    %if (i == 1)
    %    estimated_time = user_param.max_iter*toc/60;
    %    fprintf('Estimated value iteration time = %d min. \n', estimated_time)
    %end

end

% Or don't iterate
% P_reach_upper = V_upper;

end

