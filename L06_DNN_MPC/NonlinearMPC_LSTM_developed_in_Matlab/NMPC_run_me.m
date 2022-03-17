clc
clear all
close all

%% Nonlinear MPC
nx = 53;
ny = 1;
nu = 3;

nlobj = nlmpc(nx,ny,nu);

%% sampling time
Ts = 0.08;
p = 5;
m = 1;


%% Weights

nlobj.Weights.OutputVariables = [1];
nlobj.Weights.ManipulatedVariables = [0.2, 0, 0.01];
nlobj.Weights.ManipulatedVariablesRate = [0.1, 0.1, 0.1];

%% Sover options 

nlobj.Optimization.SolverOptions.Display = 'iter';
nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-10;
nlobj.Optimization.SolverOptions.StepTolerance = 1e-10;
nlobj.Optimization.SolverOptions.Algorithm = 'sqp';
nlobj.Optimization.SolverOptions.SpecifyObjectiveGradient = true;
nlobj.Optimization.SolverOptions.SpecifyConstraintGradient = true;
nlobj.Optimization.SolverOptions.MaxIterations = 5000;
%% horizon.

nlobj.Ts = Ts;
nlobj.PredictionHorizon = p; 
nlobj.ControlHorizon = m;



%% Constrain inputs
nlobj.MV(1).Min = (10-38.6868)/11.864;
nlobj.MV(1).Max = (90-38.6868)/11.864;

nlobj.MV(2).Min = (-10+3.97)/3.433;
nlobj.MV(2).Max = (3+3.97)/3.433;

nlobj.MV(3).Min = (0.8-0.902)/0.058;
nlobj.MV(3).Max = (1-0.902)/0.058;


%% contraint softenning

nlobj.Weights.ECR = 0.001;


%% StateFcn

nlobj.Model.StateFcn = 'MyLSTMstateFnc';

nlobj.Model.IsContinuousTime = false;


nlobj.Model.OutputFcn = @(x,u) [x(1)];
%% validation
% x0 = [214.6833;1.3991;557.7784;0.0263588853137028;0.130665032573534;-0.0422806116987781;-0.0675513076973337;-0.0194217642937481;0.0477548025602380;-0.0163305022317555;0.0679552790345901;0.0936930475197400;0.0466112048808459;0.113948499727606;0.181857066989532;-0.137744629904436;-0.0411360784392403;0.0288717040180779;-0.128586147062287;0.107365876624987;-0.0532273495782595;0.0399525548217938;0.138171823129510;0.0657013134784230;0.119079951626619;0.0543322471296758;-0.0775467840932733;0.0903947915669897;-0.164213133580873;-0.227440273264433;-0.141612446542211;0.110468121971166;0.144392920179593;-0.271528234257662;-0.280573314046113;-0.0740311556655427;-0.0504756509345822;-0.00446615739300279;0.241247063343754;-0.139437859508838;-0.0134093006356106;0.196665945768693;-0.287007541794395;0.00389182667781415;0.224062026037868;0.166697661738291;-0.294814669337541;-0.159823672829030;-0.164558860898588;0.0724193453704733;-0.00499028593518080;-0.174820842113629;0.0978323215690729;-0.0694348420482519;0.00439924031131990;0.0513715014987949;0.305531812257048;-0.0929270260309927;-0.138514976728746;-0.0364979939247578;0.0916627684762851;-0.0377607691520452;0.147452794634218;0.202953711809652;0.104452225537947;0.244188054375657;0.332797644879101;-0.249049868485892;-0.0798757252062737;0.0556413984582058;-0.236002559188520;0.219836536459521;-0.0963537690635108;0.0725230991961644;0.275054523437011;0.154386685745233;0.214368707909448;0.106938890721629;-0.159691328973126;0.180865646954916;-0.316480654172324;-0.545460172181690;-0.247075675991764;0.224901379567982;0.263520645672167;-0.502741601588333;-0.657576466869519;-0.115306843374142;-0.0995020540694526;-0.00921002817899062;0.480231585055435;-0.242602576487329;-0.0219928391167205;0.361812260261348;-0.520901579198180;0.00799112427376636;0.448366029877033;0.355268235656300;-0.786594412845805;-0.330252193265545;-0.276465893162674;0.140875765269575;-0.0103723208845426;-0.309422900261510;0.189600394343941;-0.128297494215035;0.0100730877569062];
x0 = [214.683300000000; 0.0263588853137028;0.130665032573534;-0.0422806116987781;-0.0675513076973337;-0.0194217642937481;0.0477548025602380;-0.0163305022317555;0.0679552790345901;0.0936930475197400;0.0466112048808459;0.113948499727606;0.181857066989532;-0.137744629904436;-0.0411360784392403;0.0288717040180779;-0.128586147062287;0.107365876624987;-0.0532273495782595;0.0399525548217938;0.138171823129510;0.0657013134784230;0.119079951626619;0.0543322471296758;-0.0775467840932733;0.0903947915669897;-0.164213133580873; 0.0513715014987949;0.305531812257048;-0.0929270260309927;-0.138514976728746;-0.0364979939247578;0.0916627684762851;-0.0377607691520452;0.147452794634218;0.202953711809652;0.104452225537947;0.244188054375657;0.332797644879101;-0.249049868485892;-0.0798757252062737;0.0556413984582058;-0.236002559188520;0.219836536459521;-0.0963537690635108;0.0725230991961644;0.275054523437011;0.154386685745233;0.214368707909448;0.106938890721629;-0.159691328973126;0.180865646954916;-0.316480654172324];
u0 = [24.6000;-3.8000;1.0000];

validateFcns(nlobj,x0,u0)

%% loadref
load_ref = RandomArray(130, 310, 100, 50);


%% Simulink

open('Load_control_NLMPC.slx')
sim('Load_control_NLMPC.slx')


