% draw figures for SNN results in Matlab


% General Network info
nSNeuron = 512; % # of neurons/units in a sensory sub-network
nSPool = 8; % # of sensory sub-networks
nRNeuron = 1024; % # of neurons/units in the second layer (Random layer)

fontSize = 16;
fontName = 'Myriad Pro';

% general figure properties
y_lim = [-5 85];
r_amps = 0:2:18;
fig_position_3x1 = [0          0         560        1313];
fig_position_2x2 = [60   167   939   630];
blackrgb = [0 0 0];
grayrgb = [0.5 0.5 0.5];
ax_name = {'box','fontsize','fontname','linewidth','tickdir','ticklength'};
ax_value = {'off',fontSize,fontName,2,'out',[.01 .01]};
ax_name_crf = {'xlim','ylim','xtick','ytick'};
ax_value_crf = {[0 19],y_lim,[0:2:20],[0:20:80]};

%% Figure 2: raster plots for sensory and attention tasks, CRFs, contrast gain plots
% load saved spike data
load('F_raster_kappa-0.mat')

% Sensory Task
thisStim = 8; % which stimulus was presented?
thisPool = 1; % which sub-network was stimulated?
S_dat = S_SensoryTask{thisStim,thisPool};
R_dat = R_SensoryTask{thisStim,thisPool};
taskDur = 300;

dot_size = 5;
dot_color = 'k';
xlims = [-taskDur/30 taskDur*1.03];
patch_color_stim = [13, 135, 95]./255;


figure(1); clf; hold all;

% Sensory Task - First layer
subplot(2,4,1);
scatter(S_dat(1,:),S_dat(2,:),dot_size,dot_color,'filled','o'); hold on;
for sp = 1:nSPool
    plot(xlims,[nSNeuron*sp,nSNeuron*sp],'k','linewidth',1); hold on;
end
xlim(xlims);
ylim([-10 nSNeuron*nSPool+10]);
xlabel('Time (ms)');
ylabel('First Layer Sub-networks');
yticks(nSNeuron/2:nSNeuron:nSNeuron*nSPool);
yticklabels(1:8);
% add patch for stimulus duration
yl = ylim;
patch([0,taskDur,taskDur,0],[yl(2),yl(2),yl(1),yl(1)],patch_color_stim,'facealpha',0.3,'linestyle','none');
set(gca,ax_name,ax_value,'yticklabelrotation',90);

% Sensory Task - Second layer
subplot(2,4,2);
scatter(R_dat(1,:),R_dat(2,:),dot_size,dot_color,'filled','o');
xlim(xlims);
ylim([-10 nRNeuron+10]);
xlabel('Time (ms)');
ylabel('Second Layer Neurons');
yticks([1,nRNeuron]);
% add patch for stimulus duration
yl = ylim;
patch([0,taskDur,taskDur,0],[yl(2),yl(2),yl(1),yl(1)],patch_color_stim,'facealpha',0.3,'linestyle','none');
set(gca,ax_name,ax_value,'yticklabelrotation',90);


% Attention Task
thisRamp = 8; % what level of attention gain?
thisSS = 6; % which stimulus strength?
thisAttStim = 1; % which of the two stimuli was attended?
taskDur = 1000;

% subset data for plotting
S_dat = S_AttentionTask{r_stim_amps==thisRamp,stim_strengths==thisSS,thisAttStim};
R_dat = R_AttentionTask{r_stim_amps==thisRamp,stim_strengths==thisSS,thisAttStim};

dot_size = 5;
dot_color = 'k';
xlims = [-taskDur/30 taskDur*1.03];
patch_color_stim = [175, 37, 8]./255;

% Attention Task - First layer
subplot(2,4,3);
scatter(S_dat(1,:),S_dat(2,:),dot_size,dot_color,'filled','o'); hold on;
for sp = 1:nSPool
    plot(xlims,[nSNeuron*sp,nSNeuron*sp],'k','linewidth',1); hold on;
end
xlim(xlims);
ylim([-10 nSNeuron*nSPool+10]);
xlabel('Time (ms)');
ylabel('First Layer Sub-networks');
yticks(nSNeuron/2:nSNeuron:nSNeuron*nSPool);
yticklabels(1:8);
% set(gca,'tickdir','none'); % can't change just for the y axis..
% add patch for task
yl = ylim;
patch([0,taskDur,taskDur,0],[yl(2),yl(2),yl(1),yl(1)],patch_color_stim,'facealpha',0.3,'linestyle','none');
set(gca,ax_name,ax_value,'yticklabelrotation',90);

% Attention Task - Second layer
subplot(2,4,4);
scatter(R_dat(1,:),R_dat(2,:),dot_size,dot_color,'filled','o');
xlim(xlims);
ylim([-10 nRNeuron+10]);
xlabel('Time (ms)');
ylabel('Second Layer Neurons');
yticks([1,nRNeuron]);
yl = ylim;
patch([0,taskDur,taskDur,0],[yl(2),yl(2),yl(1),yl(1)],patch_color_stim,'facealpha',0.3,'linestyle','none');
set(gca,ax_name,ax_value,'yticklabelrotation',90);

% CRF + cont gain
% FR for attended and unattended stim in stimulated sub-network
load('result_F_unattStim.mat');
nStimStrength = size(avg_FR_stim_att_kappa,1); stim_strengths = (1:nStimStrength) - 1;
nRAmp = size(avg_FR_stim_att_kappa,2);
nReps = size(avg_FR_stim_att_kappa,3);

colors=viridis(nRAmp); colors = flipud(colors);
kappa = 1; % actual kappa value = 0

repavg_FR_stim_att = squeeze(mean(avg_FR_stim_att_kappa(:,:,:,kappa),3));
se_FR_stim_att =  std(squeeze(avg_FR_stim_att_kappa(:,:,:,kappa)),[],3)/sqrt(nReps);

repavg_FR_stim_unatt = squeeze(mean(avg_FR_stim_unatt_kappa(:,:,:,kappa),3));
se_FR_stim_unatt = std(squeeze(avg_FR_stim_unatt_kappa(:,:,:,kappa)),[],3)/sqrt(nReps);

% CRF - Attended stimulus
subplot(2,3,4);
for r_amp = 1:nRAmp
    h1(r_amp) = plot(stim_strengths,repavg_FR_stim_att(:,r_amp),'color',colors(r_amp,:),'linewidth',2,'linestyle','-','marker','o'); hold on;
    errorAreaRR(stim_strengths,repavg_FR_stim_att(:,r_amp),se_FR_stim_att(:,r_amp),colors(r_amp,:),colors(r_amp,:)); hold on;
    leg{r_amp} = num2str(r_amps(r_amp));
end
legend(h1,leg,'location','east','box','off');
ylabel('Avg FR (Hz)');
xlabel('Stimulus Strength');
set(gca,ax_name,ax_value,ax_name_crf,ax_value_crf);
pbaspect([1 1 1]);

% CRF - Unattended stimulus
subplot(2,3,5);
for r_amp = 1:nRAmp
    h2(r_amp) = plot(stim_strengths,repavg_FR_stim_unatt(:,r_amp),'color',colors(r_amp,:),'linewidth',2,'linestyle','--','marker','x'); hold on;
    errorAreaRR(stim_strengths,repavg_FR_stim_unatt(:,r_amp),se_FR_stim_unatt(:,r_amp),colors(r_amp,:),colors(r_amp,:)); hold on;
    leg{r_amp} = num2str(r_amps(r_amp));
end
legend(h2,leg,'location','west','box','off');
ylabel('Avg FR (Hz)');
xlabel('Stimulus Strength');
set(gca,ax_name,ax_value,ax_name_crf,ax_value_crf);
pbaspect([1 1 1]);

% Contrast Gain
clear avg_cg se_cg
avg_cg(:,1,:) = mean(cont_gain_att_kappa,2);
avg_cg(:,2,:) = mean(cont_gain_unatt_kappa,2);
se_cg(:,1,:) = std(cont_gain_att_kappa,[],2)/sqrt(nReps);
se_cg(:,2,:) = std(cont_gain_unatt_kappa,[],2)/sqrt(nReps);
colors=bone(6);
subplot(2,3,6);
plot(r_amps,avg_cg(:,1,kappa),'color',colors(kappa,:),'linewidth',2,'linestyle','-','marker','o'); hold on;
plot(r_amps,avg_cg(:,2,kappa),'color',colors(kappa,:),'linewidth',2,'linestyle','--','marker','x'); hold on;
errorAreaRR(r_amps,avg_cg(:,1,kappa),se_cg(:,1,kappa),colors(kappa,:),colors(kappa,:)); hold on;
errorAreaRR(r_amps,avg_cg(:,2,kappa),se_cg(:,2,kappa),colors(kappa,:),colors(kappa,:)); hold on;
set(gca,ax_name,ax_value,'XLim',[0 max(r_amps)],'XTick',r_amps,'ylim',[0.15 0.65],'ytick',[.2:.1:.6]);
legend({'Attended','Unattended'},'box','off','location','southwest');
ylabel('Contrast Gain (a.u.)');
xlabel('Top-down Gain Strength');
pbaspect([1 1 1]);
set(gcf,'color','white','position',fig_position_2x2);

%% Figure 3: Decoding (Circular regression & SVM)
% all stim strength level collapsed
kappas = [0,0.1,0.2,0.3,0.4]; 
kp = 1; % kappa = 0 for now
load('result_F_label.mat') % grabbing label_stim_strength_main
load(['result_F_kappa-' num2str(kappas(kp)) '.mat']); % saved decoding results

nReps = numel(reps);

% calculate sen2sen MAE from decoding_sen2sen on the fly
% reporting only in text and not plotting any figure for this
% collapse over all pools and reps
tsty = decoding_sen2sen(:,:,1,:); % (trial, pool, [tst,pred], rep)
pred_deg = decoding_sen2sen(:,:,2,:);
temp = squeeze(pred_deg - tsty);
temp(abs(temp)>180) = temp(abs(temp)>180) - sign(temp(abs(temp)>180))*2*180;
cal_MAE = squeeze(mean(abs(temp),1));
MAE_sen2sen = squeeze(mean(abs(temp),1)); % sen2att (S_N_pools,nRep,len(r_stim_amps))
fprintf('MAE sen2sen: %1.3f\n',mean(MAE_sen2sen,'all'));
MAE_sen2sen = permute(MAE_sen2sen,[3,1,2]); % (len(r_stim_amps),S_N_pools,nRep)

% calculate MAE from decoding_sen2att - all trials
tsty = decoding_sen2att(:,:,1,:,:); % decoding_sen2att[trial,pool,(real,pred),rep]
pred_deg = decoding_sen2att(:,:,2,:,:);
temp = squeeze(pred_deg - tsty);
temp(abs(temp)>180) = temp(abs(temp)>180) - sign(temp(abs(temp)>180))*2*180;
MAE_sen2att = squeeze(mean(abs(temp),1)); % sen2att (S_N_pools,nRep,len(r_stim_amps))
MAE_sen2att = permute(MAE_sen2att,[3,1,2]); % (len(r_stim_amps),S_N_pools,nRep)

figure(2); clf;

% Circular regression model trained on sensory task and tested on attention task
% histograms of predicted labels
cnt = 0;
for r_stim = 1:size(decoding_sen2att,5)
    cnt = cnt+1;
    subplot(4,5,cnt);

    % real vs pred
    real_label=reshape(decoding_sen2att(:,1,1,:,r_stim),[],1);
    pred_label=reshape(decoding_sen2att(:,1,2,:,r_stim),[],1);
    h1 = histogram(pred_label(real_label==90),0:12:360,'facecolor','blue','FaceAlpha',0.6); hold on; % these edges makes sure 90 and 270 are at the center of the bars
    h2 = histogram(pred_label(real_label==270),0:12:360,'facecolor','red','FaceAlpha',0.6);
    if r_stim == 1
    legend([h1,h2],{'Attend 90','Attend 270'},'box','off','location','northwest')
    ylabel('Trial Count');
    
    end
    if r_stim == 6
        ylabel('Trial Count');
    xlabel('Predicted Stimulus');
    end
    title(['Gain: ', num2str(r_stim_amps(r_stim))],'fontweight','normal');
    set(gca,ax_name,ax_value,'yLim',[0 5000],'ytick',[0 4000],'XTick',0:90:360,...
        'xticklabel',{'0','90','180','270','360'},'yticklabelrotation',90);
end


% MAE plot
avgMAE_s2e = mean(squeeze(MAE_sen2att(:,1,:)),2); % only the stimulated sub-network
seMAE_s2e = std(squeeze(MAE_sen2att(:,1,:)),[],2)/sqrt(nReps);

avgMAE_s2e_unstim = mean(squeeze(mean(MAE_sen2att(:,2:end,:),2)),2); % unstimulated sub-networks
seMAE_s2e_unstim = std(squeeze(mean(MAE_sen2att(:,2:end,:),2)),[],2)/sqrt(nReps);

subplot(2,3,4);
clear h
h(1) = plot(r_stim_amps, avgMAE_s2e,'linewidth',2,'color','k','linestyle','-','marker','o'); hold on;
errorAreaRR(r_stim_amps,avgMAE_s2e,seMAE_s2e,blackrgb,blackrgb); hold on;

h(2) = plot(r_stim_amps, avgMAE_s2e_unstim,'linewidth',2,'color',grayrgb,'linestyle','--','marker','x'); hold on;
errorAreaRR(r_stim_amps,avgMAE_s2e_unstim,seMAE_s2e_unstim,grayrgb,grayrgb);

ylabel('MAE (deg)');
xlabel('Feature Gain Strength');
legend(h,{'Stimulated','Unstimulated'},'box','off','location','northeast');
set(gca,ax_name,ax_value,'xlim',[0 max(r_stim_amps)],'xtick',[0:2:18]);
pbaspect([1 1 1]);


% SVM decoding trained/tested on attention task
decoding_att2att = decoding_att2att.*100; % decoding_att2att[pool,ramp,rep]
avgDec = mean(squeeze(decoding_att2att(1,:,:)),2); % only the stimulated sub-network
seDec = std(squeeze(decoding_att2att(1,:,:)),[],2)/sqrt(nReps);

avgDec_unstim = mean(squeeze(mean(decoding_att2att(2:end,:,:),1)),2); % unstimulated sub-networks
seDec_unstim = std(squeeze(mean(decoding_att2att(2:end,:,:),1)),[],2)/sqrt(nReps);

subplot(2,3,5);
clear h
h(1) = plot(r_stim_amps, avgDec,'linewidth',2,'color','k','linestyle','-','marker','o'); hold on;
errorAreaRR(r_stim_amps,avgDec,seDec,blackrgb,blackrgb); hold on;

h(2) = plot(r_stim_amps, avgDec_unstim,'linewidth',2,'color',grayrgb,'linestyle','--','marker','x'); hold on;
errorAreaRR(r_stim_amps,avgDec_unstim,seDec_unstim,grayrgb,grayrgb);

ylim([45 105]);
ylabel('Decoding Accuracy (%)')
xlabel('Feature Gain Strength');
legend(h,{'Stimulated','Unstimulated'},'box','off','location','southeast');

set(gca,ax_name,ax_value,'xlim',[0 max(r_stim_amps)],'xtick',[0:2:18]);
pbaspect([1 1 1]);


% SVM decoding - cross-generalizing between unstimualted sub-networks
decoding_att2att_xpool = decoding_att2att_xpool.*100; % decoding_att2att_xpool[pool-1,ramp,rep]
avgDec_x = mean(squeeze(mean(decoding_att2att_xpool,1)),2);
seDec_x = std(squeeze(mean(decoding_att2att_xpool,1)),[],2)/sqrt(nReps);

subplot(2,3,6);
clear h
h = plot(r_stim_amps, avgDec_x,'linewidth',2,'color',grayrgb,'linestyle','--','marker','x'); hold on;
errorAreaRR(r_stim_amps,avgDec_x,seDec_x,grayrgb,grayrgb); hold on;
ylim([45 105]);
ylabel('Decoding Accuracy (%)')
xlabel('Feature Gain Strength');

set(gca,ax_name,ax_value,'xlim',[0 max(r_stim_amps)],'xtick',[0:2:18]);
pbaspect([1 1 1]);

set(gcf,'color','white','position',[60     3   939   794]);

%% Figure 4: Probability distribution as a function of K
kappas = 0:0.1:0.4;
x = linspace(0,2*pi-(2*pi/512),512);
figure(3); clf;
colors = parula(numel(kappas)+1);
clear leg h
for kk = 1:numel(kappas)
    circ_norm_func = exp(kappas(kk) * cos(x-pi)) / (2 * pi * besseli(0,kappas(kk)));
    circ_norm_func = circ_norm_func./sum(circ_norm_func); % normalize to amp = 1
    h(kk) = plot(circ_norm_func,'linewidth',2,'color',colors(kk,:)); hold on;
    leg{kk} = ['K = ',num2str(kappas(kk))];
end

xlim([-20 512+20]);
xticks([0,512/2, 512]);
xticklabels({'μ-180','μ','μ+180'});

% to get rid of exponent and display decimals as is
ylim([0.001 0.003]);
yticks([.001 .002 .003]);
yticklabels({'.001','.002','.003'});

legend(h,leg,'box','off','fontsize',16);
ylabel('Probability');
xlabel({'Preferred Feature of','First Layer Neuron'});
pbaspect([1 1 1]);
set(gca,ax_name,ax_value);
set(gcf,'color','white','position',[91   552   261   210]);


%% Figure 5: CRFs, MAEs as a function of K
kappas = [0:0.1:0.4];
figure(4); clf;

% FR for attended stim in stimulated and unstimulated sub-network
load('result_F_unstimPool.mat');
nStimStrength = size(avg_FR_stim_kappa,1); stim_strengths = (1:nStimStrength) - 1;
nRAmp = size(avg_FR_stim_kappa,2);
nSPools = size(avg_FR_stim_kappa,3);
nReps = size(avg_FR_stim_kappa,4);
nKappa = size(avg_FR_stim_kappa,5);
colors=viridis(size(avg_FR_stim_kappa,2)); colors = flipud(colors);
for kappa = 1:nKappa
    avg_FR_stim = avg_FR_stim_kappa(:,:,:,:,kappa);
    avg_FR_stim_att = squeeze(avg_FR_stim(:,:,1,:));
    avg_FR_stim_unatt = squeeze(mean(avg_FR_stim(:,:,2:end,:),3));
    repavg_FR_stim_att = mean(avg_FR_stim_att,3); % average over reps
    se_FR_stim_att = std(avg_FR_stim_att,[],3)/sqrt(nReps);
    repavg_FR_stim_unatt = mean(avg_FR_stim_unatt,3);
    se_FR_stim_unatt = std(avg_FR_stim_unatt,[],3)/sqrt(nReps);
    
    repavg_FR_stim_att = avg_FR_stim_att(:,:,1); % average over reps
    repavg_FR_stim_unatt = avg_FR_stim_unatt(:,:,1);
    
    % FR for attended stim in the stimulated sub-network
    subplot(3,5,kappa);
    clear h leg
    for r_amp = 1:nRAmp
        h(r_amp) = plot(stim_strengths,repavg_FR_stim_att(:,r_amp),'color',colors(r_amp,:),'linewidth',2,'linestyle','-','marker','o'); hold on;
        leg{r_amp} = num2str(r_amps(r_amp));
    end
    
    set(gca,ax_name,ax_value,ax_name_crf,ax_value_crf);
    xticks(0:5:20);
    pbaspect([1 1 1]);
    title(['K = ',num2str(kappas(kappa))],'fontweight','normal');
    if kappa == 1
        ylabel('Avg FR (Hz)');
        xlabel('Stimulus Strength');
    end
    if kappa == nKappa
        legend(h,leg,'location','east','box','off');
    end

    % FR for attended stim in the unstimulated sub-networks
    subplot(3,5,5+kappa);
    for r_amp = 1:nRAmp
        h(r_amp) = plot(stim_strengths,repavg_FR_stim_unatt(:,r_amp),'color',colors(r_amp,:),'linewidth',2,'linestyle','--','marker','x'); hold on;
    end
    if kappa == nKappa
        legend(h,leg,'location','east','box','off');
    end
    set(gca,ax_name,ax_value,ax_name_crf,ax_value_crf);
    xticks(0:5:20);

    pbaspect([1 1 1]);
    if kappa == 1
        ylabel('Avg FR (Hz)');
        xlabel('Stimulus Strength');
    end
end

% MAE plot
for kappa = 1:nKappa

    load(['result_F_kappa-' num2str(kappas(kappa)) '.mat']);

    % calculate MAE from decoding_sen2att - all trials
    tsty = decoding_sen2att(:,:,1,:,:);
    pred_deg = decoding_sen2att(:,:,2,:,:);
    temp = squeeze(pred_deg - tsty);
    temp(abs(temp)>180) = temp(abs(temp)>180) - sign(temp(abs(temp)>180))*2*180;
    MAE_sen2att = squeeze(mean(abs(temp),1)); % sen2att (S_N_pools,nRep,len(r_stim_amps))
    MAE_sen2att = permute(MAE_sen2att,[3,1,2]);

    avgMAE_s2e = mean(squeeze(MAE_sen2att(:,1,:)),2); % only the stimulated sub-network
    seMAE_s2e = std(squeeze(MAE_sen2att(:,1,:)),[],2)/sqrt(nReps);
    
    avgMAE_s2e_unstim = mean(squeeze(mean(MAE_sen2att(:,2:end,:),2)),2); % unstimulated sub-networks
    seMAE_s2e_unstim = std(squeeze(mean(MAE_sen2att(:,2:end,:),2)),[],2)/sqrt(nReps);
    
    subplot(3,5,10+kappa);
    clear h
    h(1) = plot(r_stim_amps, avgMAE_s2e,'linewidth',2,'color','k','linestyle','-','marker','o'); hold on;
    errorAreaRR(r_stim_amps,avgMAE_s2e,seMAE_s2e,blackrgb,blackrgb); hold on;
    
    h(2) = plot(r_stim_amps, avgMAE_s2e_unstim,'linewidth',2,'color',grayrgb,'linestyle','--','marker','x'); hold on;
    errorAreaRR(r_stim_amps,avgMAE_s2e_unstim,seMAE_s2e_unstim,grayrgb,grayrgb);
    
    if kappa == 1
    ylabel('MAE (deg)');
    legend(h,{'Stimulated','Unstimulated'},'box','off','location','southeast');
    xlabel('Feature Gain Strength');
    end
    set(gca,ax_name,ax_value,'xlim',[0 max(r_stim_amps)],'xtick',[0:5:20],'ylim',[0 105]);
    pbaspect([1 1 1]);
end

set(gcf,'color','white','position',fig_position_2x2);


