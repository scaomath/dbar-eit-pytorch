% This is the main script for reading data from the KIT4 system and
% applying the D-bar method to obtain a reconstruction for all examples.
%
% written by Andreas Hauptmann, 2017


% Choose experiment number in 1-8
ex=3;
% Choose data set in experiment, the maximum number of individual 
% measurements for each experiment is: [4 6 6 4 2 7 2 6]
ver=5;

plotFlag=true;
printFlag=true;

%% Evaluation of data
%Load measured data
eval(['load KIT4_measdata/dataMat_adj_' num2str(ex) '_' num2str(ver)]);

%Save for evaluation in change of basis
save data/KIT4_measurement U_ad10 U_ad0
        
      

%Transform adjacent measurements to ND map (separately)
comp02_ND_buildFromKIT4
%Convert to DN map
comp03_DN_build
%Solve BIE for CGO solutions
comp04_psi_BIE
%Compute scattering transform
comp05_tBIE_psi   

%Solve D-bar equation
comp06_Dbarsolve
        
        
%% Plotting of results

% plot results
if(plotFlag)
    load data/reconstruction p e t recon
    figure(2)
    clf
    pdeplot(p,e,t,'xydata',recon,'colormap','jet')
    
%     colorbar off, 
    axis equal, axis off, 
end        

%printing results
if(printFlag)
    figure(2)
    eval(['print -djpeg examples/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'])
end



