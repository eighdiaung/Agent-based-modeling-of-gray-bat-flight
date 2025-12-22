function batsS = model_O_v3(N_total,gamma,sigma,delta,IC,seed,obs_center)

% !!! NOTE: the simulation only runs until the final spawn so the last bat may only have a single time point !!!
% So the code runs N_total + 10 spawns and remove the last 10 bats to get almost complete trajectories for N_total bats


% bats are the raw trajectories of the simulation.
% batsS are the trajectories that are smoothed (to be consistent with experiemental data)
% N_total is the total number of bats to respawn.
% beta is the weight for obstacle avoidance behavior
% delta is the weight for flocking behavior
% gamma is the weight for goal behavior

rng(seed,'twister');
%% GEOMETRY OF THE ENVIRONMENT

% initialize the locations of obstacle centers
% obstacles span 11 (cross stream dir) x 4 (upstream dir) square meters rectangle which is approximately the same area as the 9_24 obstacle arrangement
% obs_center(1:9,1) = [0.5:1.36:12];
% obs_center(10:18,1) = [1.15:1.36:13];
% obs_center(1:9,2) = ones(9,1)*6;
% obs_center(10:18,2) = ones(9,1)*2;

% define the radius of obstacles
obs_rad = 0.0615;

%the location of walls in the X coordinate
wallxlocs = [0,14];

% wingspan of a bat
WS = 0.28; % meters

%% HYPERPARAMETERS for the model
N = 3; % number of agents to start with
N_total = N_total+10; % number of agents to simulate in total
s = 5; % speed of the agents (m/s)
tau = 1/60; % number of seconds per timestep (this is set to sample rate of data 1/60s)
mu = 14; % mean interval of frames for spawning a new bat 

% values in brackets are suggested just for testing the code and getting
% behaviors similar to the data
eta =   0.05; % noise (0.05)
delta =  delta;    % obs avoidance force (0.5)
gamma = gamma;
sigma = sigma;    % social turning force (0.1)

intrad_w = 0.8;   % interaction range with the walls (0.7 m obtained from data)
intrad_o = 0.5;   % interaction range with the obstacles (0.4 obtained from data)
intrad_h = 1.1;   % interaction range with conspecifics in numbers of wingspans (5 wingspans obtained from data)

%% INITIAL CONDITIONS

%initialize velocities
theta = ones([N,1])*pi/2 + eta*2*pi*(rand(N,1)-0.5); % bats have prefered heading to go straight with noise

% initialize positions
X = random(IC, N,1); % sample from the empirical distribution IC
X(X<WS) = WS;
Y = unifrnd(-6,-2, N,1);
A = sqrt((X -X').^2 + (Y -Y').^2) <= WS;
A = A - eye(size(A));

% while loop for initial positions to satisfy volume exclusion
while any(A(:))
    X = random(IC, N,1);
    X(X<WS) = WS;
    Y = unifrnd(-6,-2, N,1);
    A = sqrt((X -X').^2 + (Y -Y').^2) <= WS;
    A = A - eye(size(A));
end


% initialize respawns
r = round(exprnd(mu,N_total,1));
respawn_times = cumsum(r)+1;
respawn_train = zeros(max(respawn_times),1);
respawn_train(respawn_times) = 1;

%% PLOT THE ICS

% figure
% plot(X,Y,'o');
% hold on
% xline(wallxlocs(1),'r');
% xline(wallxlocs(2),'r');
% xline(intrad_w,'--k');
% title(0)
% hold off
% drawnow

%% MODEL

% Xsave and Ysave are just for plotting all the trajectories at the end of
% simulation. They are not saved into the file.
Xsave = [];
Ysave = [];

% ID keeps track of agents. If the agents go out of bounds of simulation
% they get reinialized and are re-IDed.
ID = [1:N]';

% bats is the matrix that will store trajectories in TIDY (time,id,x,y) format
bats = [];

for itime = 1: max(respawn_times)

    if N > 0

        theta0 = theta; % initial theta

        % normalized velocity vectors
        Vx = cos(theta);
        Vy = sin(theta);
        v = [Vx,Vy];

        %% wall attraction
        rwx1 = (wallxlocs(1)+ intrad_w)-X;  % this calculates the distance from the wall in the x direction
        rwx2 = (wallxlocs(2)- intrad_w)-X;  % this calculates the distance from the wall in the x direction
        rwx = ones(size(Y));
        rwy = ones(size(Y));    % this calculates the distance from the wall in the y direction, which is just zero is the wall is perpendicular to the y axis
        dw1 = abs(rwx1);    % we do not care the sign just the magnitude of the distance in x
        dw2 = abs(rwx2);    % we do not care the sign just the magnitude of the distance in x

        Aw1 = (dot([rwx1,zeros(size(Y))],v,2) >= 0);
        Aw2 = (dot([rwx2,zeros(size(Y))],v,2) >= 0);

        v_thig = [rwx1.*exp(-dw1/intrad_w) + rwx2.*exp(-dw2/intrad_w),rwy]; % this is the vector to the wall, here we care about the direction of rwx
        v_t = v_thig./vecnorm(v_thig,2,2);


        %% group spreading
        if N > 1
            rx = X' - X; % here we need to do Xtemp' - Xtemp to get the vector pointing in the direction towards the neighbor
            ry = Y' - Y; % same logic as line above
            d = sqrt(rx.^2 + ry.^2); % distance between the agents
            A_f = double((rx.*Vx + ry.*Vy) >= 0); % adjacency matrix for forward facing

            [~,I] = sort(d,2); % sort the distances per row
            minI = I(:,2); % take the second column which corresponds to nearest neighbor
            linearIdx = sub2ind(size(d), [1:size(d,1)]' , minI); % get linear indices of matrix d where the nearest neighbors are
            A_nn = zeros(size(d)); % initialize an adjacency matrix for nearest neighbors
            A_nn(linearIdx) = 1; % make ones for each row (focal agent) for the column (who their nearest neighbor is)

            theta_turn = sign(diag(Vy)*rx - diag(Vx)*ry) .* pi/2 .*exp(-d/intrad_h).*A_f.*A_nn; % turn away from the nearest neighbor
            theta_turn(theta_turn==0) = nan; % zero theta should only mean there is no turning so make them nans since when turning back to the vector theta=0 mean a dir.
            v_s_x = nansum(cos(theta_turn).*exp(-d),2); % nansum for velocity in x
            v_s_y = nansum(sin(theta_turn).*exp(-d),2); % nansum for velocity in y

            % renormalize
            v_s_x = v_s_x./sqrt((v_s_x.^2 + v_s_y.^2));
            v_s_y = v_s_y./sqrt((v_s_x.^2 + v_s_y.^2));
            v_s = [v_s_x,v_s_y];

            % make nan values zero
            v_s(isnan(v_s)) = 0;

        else
            v_s = [0,0];
        end

        %% avoidance to obstacles
        % loop through each obstacle to find the distance to the obstacles from each of the agent
        % at each loop matrices rox, roy, do, and thetao are populated which are of sizes (number of agents x number of obstacles)
        for iobs = 1 : size(obs_center,1)
            % X, and Y contains all the agents' positions at this current timestep
            rox(:,iobs) = obs_center(iobs,1)-X; % vector pointing from agents towards the obstacle in x
            roy(:,iobs) = obs_center(iobs,2)-Y; % vector pointing from agents towards the obstacle in y
            do(:,iobs) = sqrt(rox(:,iobs).^2 + roy(:,iobs).^2); % finding the magnitude of the distance
            thetao(:,iobs) = atan2(roy(:,iobs),rox(:,iobs)); % finding the angle in world coordinates of ro vector
        end

        % for each row in do find the minimum across the columns, so that for each agent, we know the distance to the nearest obstacle
        [~,ind] = min(do,[],2);
        rows = (1:N)';  % column vector of row indices

        % the line below converts the row and col indices into linear indices of the matrix
        linear_indices = sub2ind(size(rox), rows, ind);

        % use the linear indices to get the elements of the matrices for ro do
        % and thetao so now they are (number of agents x 1) vectors, for each
        % agent the values are for the nearest obstacle
        rox = rox(linear_indices);
        roy = roy(linear_indices);
        ro = [rox,roy];
        do = do(linear_indices);
        thetao = thetao(linear_indices);

        % the code block below is the similar logic as the wall maneuver behavior
        % the turning direction is the sign of the cross product: vector to the obstacle x velocity of agent
        turningdir = sign( rox .* Vy - roy .* Vx);
        turningdir(turningdir==0) = 2 * randi([0, 1], sum(turningdir==0), 1) - 1;
        theta_a = theta  +  (turningdir.*(pi/2.*(dot(ro,v,2) >= 0))).*exp(-do./intrad_o);  % behavior rule
        v_o = [cos(theta_a),sin(theta_a)];


        %% sum up all the behaviors
        % (1-beta-delta-gamma) is the weight for inertia
        % beta is the weight for geometry avoidance behavior
        % delta is the weight for the social behavior to match heading changes
        % gamma is the goal oriented behavior
        v = (1-gamma-sigma-delta)*v + gamma*v_t + sigma*v_s + delta*v_o;


        % renormalize the behavior sum and add noise where eta is the weight for noise
        theta = atan2(v(:,2),v(:,1)) + eta*2*pi*(rand(N,1)-0.5);


        %% preliminary update on the velocity and positions
        Vx = cos(theta);
        Vy = sin(theta);

        Xtemp = X + s*Vx*tau;
        Ytemp = Y + s*Vy*tau;

        %% check for physical violations - this check is only for specific cases when there is violation to volume exclusions with the geometry of the environment
        % the codes below are not to be confused as behavior rules
        % they are soley constraints of the existing geometry in the environment

        %% check for violations

        % violations with walls left(1) and right(2)
        violate_wall1 = Xtemp < (wallxlocs(1)+WS/2); % if the agent's X position is less than the X position of the left wall + half a wingspan of a bat
        violate_wall2 = Xtemp > (wallxlocs(2)-WS/2); % if the agent's X position is more than the X position of the right wall - half a wingspan of a bat

        % code below is to first search for the closest obstacles for each agent then to check the violation with that obstacle
        for iobs = 1 : size(obs_center,1)
            rox(:,iobs) = obs_center(iobs,1)-Xtemp;
            roy(:,iobs) = obs_center(iobs,2)-Ytemp;
            do(:,iobs) = sqrt(rox(:,iobs).^2 + roy(:,iobs).^2);
            thetao(:,iobs) = atan2(roy(:,iobs),rox(:,iobs));
        end
        [~,ind] = min(do,[],2);
        rows = (1:N)';  % column vector of row indices
        linear_indices = sub2ind(size(rox), rows, ind);
        ro = [rox(linear_indices),roy(linear_indices)];
        do = do(linear_indices);
        violate_obs = do < (obs_rad+WS/2); % check if each of the agent is less than the obstacle radius + half a wingspan of a bat

        % check for violation between agents - they should be at least 1 wingspan away
        violate_bats = sqrt((Xtemp - Xtemp').^2 + (Ytemp - Ytemp').^2) <= WS;
        violate_bats = violate_bats - eye(size(violate_bats));

        % combine all violation checks into a matrix of N x 3+N
        violation_check = [violate_wall1, violate_wall2, violate_obs, violate_bats];


        %% code below is a while loop that will make the agents turn to prevent physical violations until satisfied
        while any(violation_check(:)) % while loop runs if any element of the violation_check matrix contains a 1

            %disp("violation")

            % Note: we will just calculate for all the agents regardless of their violation.
            % Later in the code we will only apply the violations if necessary

            %% turn away from left wall due to volume violation
            rwx = wallxlocs(1)-X;
            rwy = zeros(size(Y));
            turningdir = sign( Vx.*rwy - Vy.*(-rwx) );
            turningdir(turningdir==0) = 2 * randi([0, 1], sum(turningdir==0), 1) - 1;

            % instead of just adding pi/2 we allow the agent to turn any angle [0,pi/2]
            % this is useful for particles to not get stuck if there are other
            % agents that are also around - helps them find a clear path
            theta_wall1 = theta + turningdir.*pi/2.*unifrnd(0,1,N,1);

            % here is where we apply zeros if there is no violation
            v_wall1x = cos(theta_wall1).*violate_wall1;
            v_wall1y = sin(theta_wall1).*violate_wall1;

            %% turn away from right wall due to volume violation (same logic as above)
            rwx = wallxlocs(2)-X;
            rwy = zeros(size(Y));
            turningdir = sign( Vx.*rwy - Vy.*(-rwx) );
            turningdir(turningdir==0) = 2 * randi([0, 1], sum(turningdir==0), 1) - 1;
            theta_wall2 = theta + turningdir.*pi/2.*unifrnd(0,1,N,1);
            v_wall2x = cos(theta_wall2).*violate_wall2; % here is where we apply zeros if there is no violation
            v_wall2y = sin(theta_wall2).*violate_wall2;

            %% turn away from obstacles due to volume violation
            % check for obstacle violation with closest obstacle to each agent since we do
            % not need to be checking for all obstacles (same logic as the avoidance)
            for iobs = 1 : size(obs_center,1)
                rox(:,iobs) = obs_center(iobs,1)-X;
                roy(:,iobs) = obs_center(iobs,2)-Y;
                do(:,iobs) = sqrt(rox(:,iobs).^2 + roy(:,iobs).^2);
                thetao(:,iobs) = atan2(roy(:,iobs),rox(:,iobs));
            end
            [~,ind] = min(do,[],2);
            rows = (1:N)';  % column vector of row indices
            linear_indices = sub2ind(size(rox), rows, ind);
            rox = rox(linear_indices);
            roy = roy(linear_indices);
            do = do(linear_indices);
            thetao = thetao(linear_indices);
            turningdir = sign(rox .* Vy - roy .* Vx);
            turningdir(turningdir==0) = 2 * randi([0, 1], sum(turningdir==0), 1) - 1;
            theta_o = theta + turningdir.*pi/2.*unifrnd(0,1,N,1); % same logic to turn [0,pi/2] away from obstacle
            v_obsx = cos(theta_o).*violate_obs;
            v_obsy = sin(theta_o).*violate_obs;


            %% turn away from bats due to volume violation

            % initialize the turning vectors - each row will be a turning direction of an agent due to violation with all other agents
            v_batsx = zeros(N,N);
            v_batsy = zeros(N,N);

            % check for violation
            dir_to_neighbor_x = X' - X; % here we need to do X' - X to get the vector pointing in the direction towards the neighbor
            dir_to_neighbor_y = Y' - Y; % same logic as line above

            % we will use linear indices to find agents that on top of each other
            [rows,cols] = find(violate_bats);
            linear_indices = sub2ind(size(v_batsx), rows, cols);
            turningdir = sign(dir_to_neighbor_x .* Vy - dir_to_neighbor_y .* Vx);
            theta_bats = theta.*(ones(N,N)) + turningdir.*pi/2.*unifrnd(0,1,N,N); % same logic to turn [0,pi/2]
            % we need to multiply again with the violate_bats matrix to zero out the velocities if there was no violation
            v_batsx(linear_indices) = cos(theta_bats(linear_indices));
            v_batsy(linear_indices) = sin(theta_bats(linear_indices));

            % collect all the turning directions in x and y for each agent that is violating wall1, wall2, closest obstalce, or any of N other agents (including itself but that is zeroed out)
            v_violationx = [v_wall1x,v_wall2x,v_obsx,v_batsx]; % N x 3 + N matrix
            v_violationy = [v_wall1y,v_wall2y,v_obsy,v_batsy];

            agents_that_violate = any(violation_check,2); % find all the indices out of N agents that are violating any 3 + N objects

            %% code below applies the turning due to volume exclusion (ONLY) to the agents that are in violation
            % update velocity of those that are in violation
            Vx(agents_that_violate) = sum(v_violationx(agents_that_violate,:), 2);
            Vy(agents_that_violate) = sum(v_violationy(agents_that_violate,:), 2);

            % renormalize the velocities
            Vx(agents_that_violate) = Vx(agents_that_violate)./sqrt(Vx(agents_that_violate).^2 + Vy(agents_that_violate).^2);
            Vy(agents_that_violate) = Vy(agents_that_violate)./sqrt(Vx(agents_that_violate).^2 + Vy(agents_that_violate).^2);

            % update only the positions of those that are in violation
            Xtemp(agents_that_violate) = X(agents_that_violate) + s*Vx(agents_that_violate)*tau;
            Ytemp(agents_that_violate) = Y(agents_that_violate) + s*Vy(agents_that_violate)*tau;

            theta = atan2(Vy,Vx);

            %% recheck violation after update
            violate_wall1 = Xtemp < (wallxlocs(1)+WS/2); % wall 1
            violate_wall2 = Xtemp > (wallxlocs(2)-WS/2); % wall 2
            for iobs = 1 : size(obs_center,1)
                rox(:,iobs) = obs_center(iobs,1)-Xtemp;
                roy(:,iobs) = obs_center(iobs,2)-Ytemp;
                do(:,iobs) = sqrt(rox(:,iobs).^2 + roy(:,iobs).^2);
                thetao(:,iobs) = atan2(roy(:,iobs),rox(:,iobs));
            end
            [~,ind] = min(do,[],2);
            rows = (1:N)';  % column vector of row indices
            linear_indices = sub2ind(size(rox), rows, ind);
            ro = [rox(linear_indices),roy(linear_indices)];
            do = do(linear_indices);
            violate_obs = do < (obs_rad+WS/2); % closest obstacle
            violate_bats = sqrt((Xtemp - Xtemp').^2 + (Ytemp - Ytemp').^2) <= WS;
            violate_bats = violate_bats - eye(size(violate_bats)); % bats

            % repopulate the violation check matrix which will be passed back to the while loop
            violation_check = [violate_wall1, violate_wall2, violate_obs, violate_bats];

        end

        %% after all the physical violations are checked and satisfied, compute the actual change of the agent's heading
        % heading of agent
        theta = atan2(Vy,Vx);


        %% save temporary positions
        X = Xtemp;
        Y = Ytemp;

        %% save the trajectories in TIDY format just like the experimental data
        bats = [bats; [ones(N,1)*itime, ID, X, Y]];


        %% periodic boundary condition in the Y axis. If bats leave they are taken out of the X,Y,Vx,Vy, and theta matrices
        ind = find(Y > 6.5 | Y < -6);

        if ~isempty(ind)
            X(ind) = [];
            Y(ind) = [];
            Vx(ind) = [];
            Vy(ind) = [];
            theta(ind) = [];
            ID(ind) = [];
        end

    end

    %% respawn if the respawn train = 1

    l = length(X);

    if respawn_train(itime) == 1
        X(l+1,1) = random(IC, 1,1);
        if X(l+1,1)<WS
            X(l+1,1) = WS;
        end
        Y(l+1,1) = unifrnd(-6,-2);
        check = sqrt((X(l+1,1) - X).^2 + (Y(l+1,1) - Y).^2) <=WS;
        check(l+1) = 0;

        % while loop to respawn without physical violations
        while any(check)
            X(l+1,1) = random(IC, 1,1);
            if X(l+1,1)<WS
                X(l+1,1) = WS;
            end
            Y(l+1,1) = unifrnd(-6,-2);
            check = sqrt((X(l+1,1) - X).^2 + (Y(l+1,1) - Y).^2) <=WS;
            check(l+1) = 0;
        end

        % heading direction
        theta(l+1,1) = pi/2 + eta*2*pi*(rand-0.5);
        Vx(l+1,1) = cos(theta(l+1));
        Vy(l+1,1) = sin(theta(l+1));
        ID(l+1,1) = max(bats(:,2)) + 1;
        bats = [bats; [itime, ID(l+1), X(l+1), Y(l+1)]];
    end

    %% empty variables for next loop
    rox = [];
    roy = [];
    do = [];
    thetao = [];
    N = length(X);


end

%% smooth the trajectories of bats to make fair comparison with data

batsS = [];

for i = unique(bats(:,2))'
    datax = bats(bats(:,2)==i,3);
    datay = bats(bats(:,2)==i,4);
    batsS = [batsS ; [bats(bats(:,2)==i,[1,2]) , smoothdata(datax,'movmean',5) , smoothdata(datay,'movmean',5)]];

end

%% remove the last 10 bats
bats(bats(:,2)>N_total-10,:) = [];
batsS(batsS(:,2)>N_total-10,:) = [];

end