function output_all=DOPF_timeseries(load_P,load_Q,load_id,name_master,name_parameters)
##function output_all=DOPF_singletimestep()
%% Progammer: He Yin,
%% The University of Tennessee, Knoxville
%% P,Q are the active and reactive power flow for all buses from Gadal
%% load_id is the bus number where loads are deployed
%% name_master is the name of the master.dss for the program
%% name_parameter is the .csv file name for the control parameters such as the location of the PV, Cap, and their settings



warning('off')

%% same the .dss file name


%% The PV and cap inputs are coming from IEEE123_input.csv

filename = name_parameters;
T = csvread(filename); %load the .csv file

%PV information loading
T=T(2:end,:);

PV_control_mode = T(1,3); %this parameter will determine which control strategy PV will be used, in current version it is fixed as 4
Objective_function = T(1,4); %determine the objective function, current version is fixed as minimize power loss
PV_bus = T(:,1); % this decide where to deploy the PV systems
PV_bus(PV_bus==0)=[];
PV_number = length(PV_bus);
PV_ratedpower = T(1,2); %PV penetration
%Cap information loading
Cap_bus = T(:,5); % this decide where to deploy the Capacitors systems
Cap_bus(Cap_bus==0)=[];
Cap_Phase = T(:,6);% this decide whether it is a one or three phase capacitor
Cap_Phase(Cap_Phase==0)=[];
Cap_kvar = T(:,7);% this is the rated reactive power of the capacitor
Cap_kvar(Cap_kvar==0)=[];
Cap_volt = T(:,8); % this is the voltage of the capacitor
Cap_volt(Cap_volt==0)=[];
Y_irr = T(:,9); % irradiance profile, not used in this version
Y_temp = T(:,10); % temperature profile, not used in this version


% save the parameters to local .mat file
save('PV_control.mat','PV_control_mode','Objective_function');
save('PV_profile.mat','PV_bus','PV_number','Y_irr','Y_temp','PV_ratedpower');
save('Cap_profile.mat','Cap_bus','Cap_Phase','Cap_kvar','Cap_volt');

%% prepare the load profile

 save('Load_curve.mat','load_id','load_P','load_Q','load_name');
 save('DSSname.mat','name_master','name_parameters');

%% define optimization inputs
Total_power_a = sum(load_P(1,:)); %total active power
Total_power_r = sum(load_Q(1,:)); %total reactive power
PV_power_a = PV_ratedpower*Total_power_a/PV_number/1000;
PV_power_r = PV_ratedpower*Total_power_r/PV_number/1000;
Cap_number = length(Cap_bus);

parameter_number = 0;
switch PV_control_mode
    case 1
        lb = [0.01;0.01;0.01;0.01;0.01;0.01;];
        ub = [0.99;0.99;0.99;0.99;0.99;0.99;];
        opts.InitialSwarmMatrix = [0.01 0.06 0.01 0.01 0.01 0.01];
        parameter_number = 6;
    case 2
        lb = [1.01;1.01;0.01;0.01;0.01;0.01;];
        ub = [1.99;1.99;0.99;0.99;0.99;0.99;];
        opts.InitialSwarmMatrix = [1.01 1.06 0.01 0.01 0.01 0.01];
        parameter_number = 6;
    case 3
        lb = [1.01;1.01;0.01;0.01;0.01;0.01;0.01;0.01;];
        ub = [1.99;1.99;0.99;0.99;0.99;0.99;0.99;0.99;];
        opts.InitialSwarmMatrix = [1.01 1.06 0.01 0.06 0.01 0.01 0.01 0.01];
        parameter_number = 8;
    case 4 %change to active and reactive power control: PV:both active and reactive, Cap:reactive
        lb = zeros(PV_number*2+Cap_number,1)+0;
        ub = [zeros(PV_number,1)+abs(PV_power_a);zeros(PV_number,1)+abs(PV_power_r); zeros(Cap_number,1)+600];

        opts.InitialSwarmMatrix = zeros(1,PV_number*2+Cap_number)+0.01;
        parameter_number = length(lb);
        x0 = ub./2;
    otherwise
       Disp('Unknown PV control.')
       return
end
save('constraints.mat','lb','ub');



%% start the optimization


%% start PSO
  pkg load statistics
  cf = @GA_single_VWC_nocap_IrrTemp_4_main_v2_powerloss_PS_HELICS;
  nr_variables = parameter_number;                       % Number of variables unknown (part of the decision)
  variable_size = [1 nr_variables];        % Vector representation
  var_min = 0;                          % Lower bound of decision space
  var_max = 1;                           % Upper bound of decision space

  %% Parameter Adjustment
  %#### TODO:Change max_iterations to 100 and swarm_size to 50.
  max_iterations = 2;                   % Maximum iterations in PSO algorithm
  swarm_size = 2;%50;                        % Swarm size (number of particles)
  w = 1;                                  % Inertia coefficient
  w_damp = 0.90;                          % damping of inertia coefficient, lower = faster damping
  c1 = 1;                                 % Cognitive acceleration coefficient (c1 + c2 = 4)
  c2 = 3;                                 % Social acceleration coefficient (c1 + c2 = 4)

  %% Init
  template_particle.position = [];
  template_particle.velocity = [];
  template_particle.cost = 0;
  template_particle.best.position = [];   % Local best
  template_particle.best.cost = inf;       % Local best

  % Copy and put the particle into a matrix
  particles = repmat(template_particle, swarm_size, 1);

  % Initialize global best (current worst value, inf for minimization, -inf for maximization)
  global_best.cost = inf;

  for i=1:swarm_size

    % Initialize all particles with random position inside the search space
    particles(i).position = unifrnd(var_min+var_max/4, var_max/2, variable_size);%x0';%unifrnd(var_min, var_max, variable_size);

    % Initiliaze velocity to the 0 vector
    particles(i).velocity = zeros(variable_size);

    % Evaluate the current cost
    particles(i).cost = cf(particles(i).position);

    % Update the local best to the current location
    particles(i).best.position = particles(i).position;
    particles(i).best.cost = particles(i).cost;

    % Update global best
    if (particles(i).best.cost < global_best.cost)
      global_best.position = particles(i).best.position;
      global_best.cost = particles(i).best.cost;
    endif

  endfor

  % Best cost at each iteration
  best_costs = zeros(max_iterations, 1);


  %% PSO Loop
  initial_time = time();
  for iteration=1:max_iterations

    for i=1:swarm_size

      % Initialize two random vectors
      r1 = rand(variable_size);
      r2 = rand(variable_size);

      % Update velocity
      particles(i).velocity = (w * particles(i).velocity) ...
        + (c1 * r1 .* (particles(i).best.position - particles(i).position)) ...
        + (c2 * r2 .* (global_best.position - particles(i).position));

      % Update position
      particles(i).position = particles(i).position + particles(i).velocity;

      % Update cost
      particles(i).cost = cf(particles(i).position);

      % Update local best (and maybe global best) if current cost is better
      if (particles(i).cost < particles(i).best.cost)
        particles(i).best.position = particles(i).position;
        particles(i).best.cost = particles(i).cost;

        % Update global best
        if (particles(i).best.cost < global_best.cost)
          global_best.position = particles(i).best.position
          global_best.cost = particles(i).best.cost
        endif

      endif
      current_time = time() - initial_time
      if current_time >=1200
        break;
      endif
    endfor

    % Get best value
    best_costs(iteration) = global_best.cost;
    best_position(iteration,:) = global_best.position;
    % Display information for this iteration
    % disp(["Iteration " num2str(iteration) ": best cost = " num2str(best_costs(iteration))]);

    % Damp w
    w = w * w_damp;

  endfor

  %% Print results
  ["Best cost: " num2str(global_best.cost)]

  %% Plot results
  figure;
  plot(best_costs, "LineWidth", 2);
  xlabel("iteration");
  ylabel("best cost");

    figure;
  plot(best_position(iteration,:),'*', "LineWidth", 2);
  xlabel("X");
  ylabel("Best Position");
  x=best_position(iteration,:);
##save('Optimization_result.mat','x','best_costs');


output_all.x = x;
output_all.best_costs = best_costs;



endfunction

function output_all=GA_single_VWC_nocap_IrrTemp_4_main_v2_powerloss_PS_HELICS(x)

%parameter loading
VWC_y_all=x;
load PV_control.mat
load PV_profile.mat
output_all=[];
switch PV_control_mode
    case 1 %VVC
        VVC_y_1 =VWC_y_all(1);
        VVC_y_2 = VWC_y_all(2);
        VWC_y_1 =1.2;
        VWC_y_2 = 1.8;
        Cap1 =VWC_y_all(3);
        Cap2 = VWC_y_all(4);
        Cap3 = VWC_y_all(5);
        Cap4 = VWC_y_all(6);
    case 2 %VWC
        VVC_y_1 = 0.2;
        VVC_y_2 = 0.8;
        VWC_y_1 =VWC_y_all(1);
        VWC_y_2 = VWC_y_all(2);
        Cap1 =VWC_y_all(3);
        Cap2 = VWC_y_all(4);
        Cap3 = VWC_y_all(5);
        Cap4 = VWC_y_all(6);
    case 3 %VVC+VWC
        VWC_y_all(1) = VWC_y_all(1) +1;
        VWC_y_all(2) = VWC_y_all(2) +1;
        VWC_y_1 =VWC_y_all(1);
        VWC_y_2 = VWC_y_all(2);
        VVC_y_1 =VWC_y_all(3);
        VVC_y_2 = VWC_y_all(4);
        Cap1 =VWC_y_all(5);
        Cap2 = VWC_y_all(6);
        Cap3 = VWC_y_all(7);
        Cap4 = VWC_y_all(8);
    case 4
        VWC_y_1 =VWC_y_all(1:PV_number);
        VWC_y_2 = 0;
        VVC_y_1 =VWC_y_all(PV_number+1:PV_number*2);
        VVC_y_2 = 0;
        Cap1 =VWC_y_all(end-3);
        Cap2 = VWC_y_all(end-2);
        Cap3 = VWC_y_all(end-1);
        Cap4 = VWC_y_all(end);
    otherwise
        VWC_y_1 =1.2;
        VWC_y_2 = 1.8;
        VVC_y_1 =0.2;
        VVC_y_2 = 0.8;
        Cap1 =VWC_y_all(1);
        Cap2 = VWC_y_all(2);
        Cap3 = VWC_y_all(3);
        Cap4 = VWC_y_all(4);
end

Cap_enable(1) = Cap1;
Cap_enable(2) = Cap2;
Cap_enable(3) = Cap3;
Cap_enable(4) = Cap4;

load PV_control.mat
load PV_profile.mat
load Cap_profile.mat
load DSSname.mat
load Load_curve.mat

Total_power = sum(sum(abs(load_P)))/96 + sum(sum(abs(load_Q)))/96;
PV_power = Total_power*(PV_ratedpower/PV_number+0.001); %1% to 30% is a reasonable range

% start to call the OpenDSS in the following function
%%---TODO Ajay/Srikanth will convert this following function
%% descriptions for the output variables
% loss: 1x1(double) the active circuit loss from the current power system
% PV_current_output: 1x14 (double) voltage and current measurement from PV buses
% PV_powera_output: 1x14 (double)active power flow from the PV buses
% PV_powerr_output: 1x14 (double) reactive power flow from the PV buses
% L_current_output: 1x98 (double) voltage and current measurement from load buses
% L_powera_output: 1x98 (double)active power flow from the Load buses
% L_powerr_output: 1x98 (double)reactive power flow from the Load buses
% solve_converged: 1x1(double) whether the power flow is converged

%% description for the input variables
% PV_number: 1x1(double) how many PVs deployed in the system
% PV_bus: the buses where PVs are depployed
% Cap_enable: 1x4(double) reactive power control for the Capacitors
% VWC_y_1: 1x14(double) active power control for the PVs
% VVC_y_1: 1x14(double) reactive power control for the PVs

%% loading local definitions
##load Load_curve.mat



Buses=[];
Caps=[];
Generators=[];
Lines=[];
Transormers=[];
PV_current_output=[];
PV_powera_output=[];
PV_powerr_output=[];
L_current_output=[];
L_powera_output=[];
L_powerr_output=[];

%% start OpenDSS here
inputs = [num2str(VWC_y_1)  num2str(VWC_y_2)  num2str(VVC_y_1)  num2str(VVC_y_2)];

Octave_to_Python_Inputs_data = [VWC_y_1  VWC_y_2  VVC_y_1  VVC_y_2]
fid = fopen('/home/run/dopf_ornl/data/Octave_to_Python_Inputs_data.txt', 'w+');
for i=1:size(Octave_to_Python_Inputs_data, 1)
    fprintf(fid, '%f ', Octave_to_Python_Inputs_data(i,:));
    fprintf(fid, '\n');
end
fclose(fid);

% solve the power flow by OpenDSS
#solve_converged = python("/home/run/dopf_ornl/DOPF_timeseries_123Bus.py", inputs);
solve_converged = python("DOPF_timeseries_123Bus.py", inputs);
% solve_converged

% Read the OpenDSS outputs
PV_current_output = readNPY('PV_current_output.npy');
PV_powera_output = readNPY('PV_powera_output.npy');
PV_powerr_output = readNPY('PV_powerr_output.npy');

L_current_output = readNPY('L_current_output.npy');
L_powera_output = readNPY('L_powera_output.npy');
L_powerr_output = readNPY('L_powerr_output.npy');

loss = readNPY('loss.npy')

% check the bus voltage see if they are within the range
voltage_diff =0;
output_all=0;
thre=0.2;%0.2; %20% voltage dynamic
v_bus=[];
nor_V = 2400;
for i=1:PV_number

    temp_ref = nor_V;
    if temp_ref>100
        temp_v=PV_current_output(i)/temp_ref-1;
        v_bus=[v_bus;temp_v]; %remove for single objective test
        if temp_v(1)>-0.5
            if max(abs(temp_v))>thre
                output_all=10000;
            end
        end
    end

            if solve_converged==0
                output_all=10000;
            end

    mean_pa = mean(abs(PV_powera_output(i)));%+abs(PV_powerr_output(i)));
    mean_all_p(i) = mean(mean_pa);

end



for i=1:length(L_current_output)

    temp_ref = nor_V;
    if temp_ref>100
        temp_v=L_current_output(i)/temp_ref-1;
            v_bus=[v_bus;temp_v];
        if ~isempty(temp_v)
            if temp_v(1)>-0.9
                if max(abs(temp_v))>thre
                    output_all=10000;
                end
            end
        end
    end

end

%% calculate the objective function
mean_p = mean(mean_all_p);
voltage_diff = voltage_diff/10;
% output_all = [mean_voltage,mean_p*1000];
output2 = mean(abs(v_bus));
w_volt = 0;%5;%0.5*10;
w_loss = 0.000001; %0.5
w_act = 0;%0.1;
load constraints.mat
for i=1:length(VWC_y_all)
##  if VWC_y_all(i)>ub(i) || VWC_y_all(i)<lb(i)
##    output_all = 10000;
##  endif

  if VWC_y_all(i)<lb(i)
    output_all = 10000;
  endif

    if VWC_y_all(i)>ub(i)
    output_all = 10000;
  endif


end

##  if sum(VWC_y_all(1:))>
##    output_all = 10000;
##  endif
if output_all~=10000
    if Objective_function == 1   % cost objective select 1
        output_all = mean_p*w_loss + output2*w_volt;  % obj of min loss and volt
%        output_all = loss*w_loss + output2*w_volt-mean_p*w_act;  % with additional obj of max PV output
    else
        output_all = -mean_p*w_act + output2*w_volt;
    end
end

endfunction



