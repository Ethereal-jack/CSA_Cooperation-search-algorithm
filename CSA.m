%  Cooperation search algorithm (CSA)

% Developed in MATLAB R2020b   

% Read the following publication first and cite if you use it

% Based on a lot of experiments, users only need to define two parameters (the number of individuals and Max Iteration)
% while three other parameters should be set as the default values for fair comparisons in their problems,
% i.e  "gbestNum=3, alpha=0.10, beta= 0.15; otherwise, it may be unfair for the CSA method",

%   title={Cooperation search algorithm: A novel metaheuristic evolutionary intelligence algorithm for numerical optimization and engineering optimization problems},
%   author={Zhong-kai Feng, Wen-jing Niu, Shuai Liu},
%   journal={Applied Soft Computing Journal 98 (2021) 106734},
%   doi={https://doi.org/10.1016/j.asoc.2020.106734}
%   publisher={Elsevier},
%   url = {https://www.sciencedirect.com/science/article/pii/S1568494620306724}
%-------------------------------------------------------------------------------------------------
function [Destination_fitness,Destination_position,Convergence_curve]=CSA(pop,gbestNum,Max_iteration,lb,ub,dim,fobj)

clear all
close all
if nargin<1
    pop=50;
    gbestNum=3;
    Max_iteration=1000;
    [lb,ub,dim,fobj]=Get_Functions();
end

display('Optimizing...');

%------------Team building phase.%Eq.(2)
current_X=rand(pop,dim).*(ub-lb)+lb;
current_X_fitness = zeros(1,pop);
% Calculate the fitness of the first set
for i=1:size(current_X,1)
    current_X_fitness(1,i)=fobj(current_X(i,:));
end

global_Best_position=zeros(gbestNum,dim);% global_Best_position
global_Best_fitness=inf(1,gbestNum);

Destination_position=zeros(1,dim);%Destination_position
Destination_fitness=inf;

Person_Best_position=current_X;%Person_Best_position
Person_Best_fitness=current_X_fitness;

[~,index_sorted]=sort(current_X_fitness);
for i=1:gbestNum
    global_Best_position(i,:)=current_X(index_sorted(i),:);
    global_Best_fitness(i)=current_X_fitness(1,index_sorted(i));
end

%Main loop
iter=1;
while iter<=Max_iteration
    % ---------------- Team Communication operator
    ave_Pbest=mean(Person_Best_position,1);
    ave_gbest=mean(global_Best_position,1);
    for i=1:size(current_X,1)
        for j=1:size(current_X,2)
            alpha = 0.10;%alpha
            beta = 0.15;%beta
            %------------Eq.(4)
            num_AK=log(1.0/Phi(0,1))*(global_Best_position(randi(gbestNum),j)-current_X(i,j));
            %------------Eq.(5)
            num_BK=alpha*Phi(0,1)*(ave_gbest(j)-current_X(i,j));
            %------------Eq.(6)
            num_CK=beta*Phi(0,1)*(ave_Pbest(j)-current_X(i,j));
            %------------Eq.(3)
            next_X(i,j)=current_X(i,j)+num_AK+num_BK+num_CK;         
        end     
        %boundary check
        Flag4ub=next_X(i,:)>ub;
        Flag4lb=next_X(i,:)<lb;
        next_X(i,:)=(next_X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end
    
    % -------Reflective Learning Operator
    reflect_X=zeros(pop,dim);
    for j=1:size(current_X,2)
        for i=1:size(current_X,1)
            if size(ub,2)==1
                ub=ones(1,dim).*ub;
                lb=ones(1,dim).*lb;
            end
            %------------Eq.(10)
            num_C= (ub(j) + lb(j)) * 0.5;
            gailv=abs(num_C- next_X(i,j))/(ub(j) - lb(j));
            if  next_X(i,j)>=num_C  %------------Eq.(7) First
                if gailv<Phi(0,1)
                    reflect_X(i,j)= Phi((ub(j) + lb(j))- next_X(i,j),num_C)  ;%------------Eq.(8) First
                else
                    reflect_X(i,j)= Phi(lb(j),(ub(j) + lb(j))- next_X(i,j))  ;%------------Eq.(8) Second
                end
            else%------------Eq.(7) Second
                if gailv<Phi(0,1)
                    reflect_X(i,j)= Phi(num_C,(ub(j) + lb(j))- next_X(i,j))  ;%------------Eq.(9) First
                else
                    reflect_X(i,j)= Phi((ub(j) + lb(j))- next_X(i,j),ub(j))  ;%------------Eq.(9) Second
                end
            end          
        end % end j
    end% end i
    
    %boundary check
    for i=1:size(current_X,1)
        Flag4ub=reflect_X(i,:)>ub;
        Flag4lb=reflect_X(i,:)<lb;
        reflect_X(i,:)=(reflect_X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end
    
    %------------Internal competition operator.  %Eq.(11)
    for i=1:size(current_X,1)
        reflect_X_fit=fobj( reflect_X(i,:));
        next_X_fit=fobj( next_X(i,:));
        if next_X_fit>reflect_X_fit
            current_X(i,:)=reflect_X(i,:);
        else
            current_X(i,:)=next_X(i,:);
        end
    end
     
    %% update fitness
    for i=1:size(current_X,1)
        current_X_fitness(1,i)=fobj(current_X(i,:));
        % Update the pbest
        if current_X_fitness(1,i)<Person_Best_fitness(1,i)
            Person_Best_fitness(1,i)=current_X_fitness(1,i);
            Person_Best_position(i,:)=current_X(i,:);
        end
    end
    
    % Update the global_Best_position
    tem_X=zeros(gbestNum*2,dim);
    tem_X_fitness=inf(1,gbestNum*2);
    [~,index_sorted]=sort(current_X_fitness);
    for i=1:gbestNum
        tem_X(i,:)=current_X(index_sorted(i),:);
        tem_X_fitness(i)=current_X_fitness(index_sorted(i));
    end
    
    tem_X=[tem_X;global_Best_position];
    tem_X_fitness=[tem_X_fitness,global_Best_fitness];
    [~,index_sorted]=sort(tem_X_fitness);
    global_Best_position=tem_X(index_sorted([1:gbestNum]),:);
    global_Best_fitness=tem_X_fitness(index_sorted([1:gbestNum]));
    [~,index_sorted]=sort(global_Best_fitness);
    
    %Convergence_curve
    Destination_position=global_Best_position(index_sorted(1),:);
    Destination_fitness=global_Best_fitness(index_sorted(1));
    display(['At iteration ', num2str(iter), ' the optimum is ', num2str(Destination_fitness)]); 
    iter=iter+1;
end

display(['The best solution obtained by CSA is ', num2str(Destination_position)]);
display(['The best optimal value of the objective funciton found by CSA is  ', num2str(Destination_fitness)]);
end

%   function
function o=Phi(num1,num2)
if num1<num2
    o=num1+rand()*abs(num2-num1);
else
    o=num2+rand()*abs(num2-num1);
end
end

%   function
function [lb,ub,dim,fobj]=Get_Functions()
lb=-100;
ub=100;
dim=30;
fobj=@Function;
end

function o=Function(x) %  function detail
o=sum(x.^2);
end