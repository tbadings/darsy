function [P_reach_lower, P_reach_upper, sigma] = robust_max_reach_probability_2layer(user_param, initial_condition, solution, solver_opts)
% Solve the maximal reachability probability problem
% Currently it only uses lower bound to compute the optimal strategy

max_iter = user_param.max_iter;
tol_VI = solver_opts.tol_VI;
Ns = user_param.Ns;
Na = user_param.Na;


% Initialize interval value
V_lower = initial_condition.V_lower;
V_upper = initial_condition.V_upper;
sigma = initial_condition.sigma;
V1_lower = zeros(user_param.Ns+1,1);
V1_upper = zeros(user_param.Ns+1,1);
V1_lower_s_a = zeros(Na,1);

% If state s \in S_terminal (unsafe)
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
        % Obtain (infty norm-based) distance by multiplying the number
        % of cells by the corresponding Delta
        for a = 1:Na
%             s
%             a
%             if ((s == 8) && (a == 2))
%                 100
%             end
            Mass = 1;
            % Order interval values
            V_lower_proj_Sp = V_lower(solution.Sp{s,a});
            gamma = solution.P_lower{s,a}';
            gamma_coarse = zeros(1, length(solution.Qp{s,a}) );
            if (any(gamma))
                for spi_unordered = 1:length(solution.Sp{s,a})
                    sp = solution.Sp{s,a}(spi_unordered);
                    qp = solution.parents{sp};
                    if ( ~isempty(qp) )
                        qpi = find(solution.Qp{s,a} == qp);
                        gamma_coarse(qpi) = gamma_coarse(qpi) + gamma(spi_unordered);
                    end
                    Mass = Mass - gamma(spi_unordered);
                end
            end
            [~, order_V_lower] = sortrows(V_lower_proj_Sp, 'ascend'); % Vector that maps the spi in variables_for-matrices to the spi in the ordered set
            for qpi = 1:length(solution.Qp{s,a})
                mass = solution.P_lower_coarse{s,a}(qpi) - gamma_coarse(qpi);
                if (mass > 0 )
                    qp = solution.Qp{s,a}(qpi);
                    if ( ~isempty(qp) )
                        children = solution.children{qp} ;
                        [~, order_children] = sortrows(V_lower(children), 'ascend');
                        children_ordered = children(order_children);
                        for sp = children_ordered'
                            spi = find(solution.Sp{s,a} == sp);
                            if (~isempty(spi))
                                g = min(mass, solution.P_upper{s,a}(spi) - gamma(spi));
                                gamma(spi) = gamma(spi) + g;
                                gamma_coarse(qpi) = gamma_coarse(qpi) + g;
                                mass = mass - g;
                                Mass = Mass - g;
                                if (mass == 0)
                                    break;
                                end
                            end
                        end
                    end
                end
%                 if (mass ~= 0)
%                     disp('ERROR: Not all mass has been assigned...')
%                 end
            end
            for spi_unordered = 1:length(solution.Sp{s,a})
                spi = order_V_lower(spi_unordered);
                sp = solution.Sp{s,a}(spi);
                qp = solution.parents{sp};
                if ( ~isempty(qp) )
                    qpi = find(solution.Qp{s,a} == qp);
                    mass = min([Mass, solution.P_upper_coarse{s,a}(qpi)-gamma_coarse(qpi), solution.P_upper{s,a}(spi)-gamma(spi)] );
                    gamma_coarse(qpi) = gamma_coarse(qpi) + mass;
                else
                    mass = min([Mass, solution.P_upper{s,a}(spi)-gamma(spi)] );
                end
                gamma(spi) = gamma(spi) + mass;
                Mass = Mass - mass;
                if (Mass == 0)
                    break;
                end
            end
            if (Mass > 0)
                disp('Error: not all mass assigned');
            end
            V1_lower_s_a(a) = gamma*V_lower_proj_Sp;
%             if ((s == 17) && (a == 8))
%                 V1_lower_q_a(a)
%             end
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
        % Obtain (infty norm-based) distance by multiplying the number
        % of cells by the corresponding Delta
        a = sigma(s);
        %             s
        %             a
        %             if ((s == 8) && (a == 2))
        %                 100
        %             end
        Mass = 1;
        % Order interval values
        V_upper_proj_Sp = V_upper(solution.Sp{s,a});
        gamma = solution.P_lower{s,a}';
        gamma_coarse = zeros(1, length(solution.Qp{s,a}) );
        if (any(gamma))
            for spi_unordered = 1:length(solution.Sp{s,a})
                sp = solution.Sp{s,a}(spi_unordered);
                qp = solution.parents{sp};
                if ( ~isempty(qp) )
                    qpi = find(solution.Qp{s,a} == qp);
                    gamma_coarse(qpi) = gamma_coarse(qpi) + gamma(spi_unordered);
                end
                Mass = Mass - gamma(spi_unordered);
            end
        end
        [~, order_V_upper] = sortrows(V_upper_proj_Sp, 'descend'); % Vector that maps the spi in variables_for-matrices to the spi in the ordered set
        for qpi = 1:length(solution.Qp{s,a})
            mass = solution.P_lower_coarse{s,a}(qpi) - gamma_coarse(qpi);
            if (mass > 0 )
                qp = solution.Qp{s,a}(qpi);
                if ( ~isempty(qp) )
                    children = solution.children{qp} ;
                    [~, order_children] = sortrows(V_upper(children), 'descend');
                    children_ordered = children(order_children);
                    for sp = children_ordered'
                        spi = find(solution.Sp{s,a} == sp);
                        if (~isempty(spi))
                            g = min(mass, solution.P_upper{s,a}(spi) - gamma(spi));
                            gamma(spi) = gamma(spi) + g;
                            gamma_coarse(qpi) = gamma_coarse(qpi) + g;
                            mass = mass - g;
                            Mass = Mass - g;
                            if (mass == 0)
                                break;
                            end
                        end
                    end
                end
            end
            %                 if (mass ~= 0)
            %                     disp('ERROR: Not all mass has been assigned...')
            %                 end
        end
        for spi_unordered = 1:length(solution.Sp{s,a})
            spi = order_V_upper(spi_unordered);
            sp = solution.Sp{s,a}(spi);
            qp = solution.parents{sp};
            if ( ~isempty(qp) )
                qpi = find(solution.Qp{s,a} == qp);
                mass = min([Mass, solution.P_upper_coarse{s,a}(qpi)-gamma_coarse(qpi), solution.P_upper{s,a}(spi)-gamma(spi)] );
                gamma_coarse(qpi) = gamma_coarse(qpi) + mass;
            else
                mass = min([Mass, solution.P_upper{s,a}(spi)-gamma(spi)] );
            end
            gamma(spi) = gamma(spi) + mass;
            Mass = Mass - mass;
            if (Mass == 0)
                break;
            end
        end
        if (Mass > 0)
            disp('Error: not all mass assigned');
        end
        V1_upper(s) = gamma*V_upper_proj_Sp;
        %             if ((s == 17) && (a == 8))
        %                 V1_upper_q_a(a)
        %             end
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

