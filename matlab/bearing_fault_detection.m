%% Bearing Fault Detection — Envelope & Order Analysis (MATLAB)
% Machine Health Monitoring & Fault Diagnosis System
% ---------------------------------------------------------
% Performs:
%   1. Bearing fault frequency calculation (BPFO, BPFI, BSF, FTF)
%   2. Band-pass filter → Hilbert envelope
%   3. Envelope spectrum with fault tone identification
%   4. Order analysis (speed-normalised spectrum)
%   5. Fault severity scoring (0–100)
%   6. Decision: Healthy / Warning / Fault
%
% Input:  CSV of raw vibration data (single channel) in ../data/raw/
% Output: diagnosis struct, plots, CSV summary in ../data/processed/
%
% Usage:
%   Run section by section or: bearing_fault_detection
%   Modify the BEARING and OPERATING parameters below for your machine.

clear; clc; close all;

%% ── Parameters ──────────────────────────────────────────────────────────────

% Acquisition
Fs       = 25600;        % Sampling rate [Hz]
data_file = fullfile('..', 'data', 'raw', 'vibration_data.csv');

% Bearing geometry — SKF 6205-2RS (change to your bearing)
n_balls        = 9;
ball_dia_mm    = 7.94;
pitch_dia_mm   = 38.50;
contact_angle  = 0.0;    % degrees (deep-groove)

% Operating condition
shaft_rpm  = 1500;       % [RPM]
shaft_hz   = shaft_rpm / 60;

% Band-pass filter for envelope analysis (around resonance)
bp_low_hz  = 2000;
bp_high_hz = 8000;

% Fault detection thresholds
KURTOSIS_WARN   = 4.0;   % kurtosis warning level
KURTOSIS_FAULT  = 6.0;   % kurtosis fault level
CF_WARN         = 5.0;   % crest factor warning
CF_FAULT        = 8.0;   % crest factor fault
TONE_AMP_THRESH = 0.01;  % min amplitude to count as a fault tone

fprintf('=================================================================\n');
fprintf('  Bearing Fault Detection — MATLAB\n');
fprintf('  Shaft: %.0f RPM  |  Fs: %.0f Hz  |  Bearing: 6205\n', shaft_rpm, Fs);
fprintf('=================================================================\n\n');


%% ── 1. Bearing Fault Frequencies ────────────────────────────────────────────

ratio = (ball_dia_mm / pitch_dia_mm) * cosd(contact_angle);

BPFO = shaft_hz * (n_balls/2) * (1 - ratio);
BPFI = shaft_hz * (n_balls/2) * (1 + ratio);
BSF  = shaft_hz * (pitch_dia_mm/(2*ball_dia_mm)) * (1 - ratio^2);
FTF  = shaft_hz * (1 - ratio) / 2;

fprintf('Bearing Fault Frequencies @ %.0f RPM:\n', shaft_rpm);
fprintf('  Shaft (1X) : %7.2f Hz  (%.2fX)\n', shaft_hz,  1.0);
fprintf('  BPFO       : %7.2f Hz  (%.2fX)\n', BPFO, BPFO/shaft_hz);
fprintf('  BPFI       : %7.2f Hz  (%.2fX)\n', BPFI, BPFI/shaft_hz);
fprintf('  BSF        : %7.2f Hz  (%.2fX)\n', BSF,  BSF/shaft_hz);
fprintf('  FTF (cage) : %7.2f Hz  (%.2fX)\n', FTF,  FTF/shaft_hz);
fprintf('\n');

fault_freqs = struct('BPFO',BPFO,'BPFI',BPFI,'BSF',BSF,'FTF',FTF,'shaft',shaft_hz);


%% ── 2. Load / Generate Signal ────────────────────────────────────────────────

if isfile(data_file)
    raw  = readmatrix(data_file);
    sig  = raw(:, 1);    % use first column
    fprintf('[OK]   Loaded signal from %s (%d samples)\n', data_file, numel(sig));
else
    fprintf('[INFO] Data file not found — using synthetic outer-race fault signal.\n');
    N    = Fs * 2;
    t_s  = (0:N-1)' / Fs;
    sig  = sin(2*pi*shaft_hz*t_s) + 0.3*sin(2*pi*2*shaft_hz*t_s);
    % Inject BPFO impulses
    impulse_times = 0 : 1/BPFO : t_s(end);
    for ti = impulse_times
        idx = round(ti*Fs) + 1;
        if idx > 0 && idx+30 <= N
            sig(idx:idx+30) = sig(idx:idx+30) + 4.0 * exp(-linspace(0,8,31)');
        end
    end
    sig = sig + 0.05*randn(N,1);
end

N  = numel(sig);
t  = (0:N-1)' / Fs;


%% ── 3. Time-Domain Features ──────────────────────────────────────────────────

rms_val      = rms(sig);
kurt_val     = kurtosis(sig);
peak_val     = max(abs(sig));
pp_val       = max(sig) - min(sig);
cf_val       = peak_val / rms_val;
skew_val     = skewness(sig);

fprintf('Time-Domain Features:\n');
fprintf('  RMS          : %.4f\n', rms_val);
fprintf('  Kurtosis     : %.4f\n', kurt_val);
fprintf('  Crest Factor : %.4f\n', cf_val);
fprintf('  Peak-to-Peak : %.4f\n', pp_val);
fprintf('  Skewness     : %.4f\n', skew_val);


%% ── 4. FFT Spectrum ──────────────────────────────────────────────────────────

NFFT  = 2^nextpow2(N);
Y     = fft(sig, NFFT);
P2    = abs(Y/N);
P1    = P2(1:NFFT/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f_axis = Fs * (0:NFFT/2) / NFFT;

[peak_amp, peak_idx] = max(P1);
peak_freq = f_axis(peak_idx);
fprintf('\nFFT: dominant frequency = %.1f Hz  (amplitude = %.4f)\n', peak_freq, peak_amp);


%% ── 5. Envelope Analysis ──────────────────────────────────────────────────────

% Band-pass filter
[b_bp, a_bp] = butter(4, [bp_low_hz, bp_high_hz]/(Fs/2), 'bandpass');
sig_bp = filtfilt(b_bp, a_bp, sig);

% Hilbert envelope
env     = abs(hilbert(sig_bp));
env_dc  = env - mean(env);    % remove DC

% Envelope spectrum
Y_env  = fft(env_dc, NFFT);
P_env  = abs(Y_env/N);
P_env  = P_env(1:NFFT/2+1);
P_env(2:end-1) = 2*P_env(2:end-1);


%% ── 6. Fault Tone Detection ──────────────────────────────────────────────────

tolerance_pct = 5;   % ±5% frequency tolerance
n_harmonics   = 3;   % check up to 3X harmonics

fault_names = {'BPFO','BPFI','BSF','FTF','shaft'};
fault_hz    = [BPFO, BPFI, BSF, FTF, shaft_hz];

detected_tones = struct('name',{},'harmonic',{},'target_hz',{},'found_hz',{},'amplitude',{});
tone_count = struct('BPFO',0,'BPFI',0,'BSF',0,'FTF',0,'shaft',0);

fprintf('\nFault Tone Detection (envelope spectrum, tolerance=%.0f%%):\n', tolerance_pct);

for fi = 1:numel(fault_hz)
    base = fault_hz(fi);
    fname = fault_names{fi};
    for h = 1:n_harmonics
        target = base * h;
        tol    = target * tolerance_pct / 100;
        mask   = f_axis >= (target-tol) & f_axis <= (target+tol);
        if any(mask)
            [max_amp, local_idx] = max(P_env(mask));
            all_idx = find(mask);
            found_hz = f_axis(all_idx(local_idx));
            if max_amp >= TONE_AMP_THRESH
                n = numel(detected_tones) + 1;
                detected_tones(n).name      = fname;
                detected_tones(n).harmonic  = h;
                detected_tones(n).target_hz = target;
                detected_tones(n).found_hz  = found_hz;
                detected_tones(n).amplitude = max_amp;
                tone_count.(fname) = tone_count.(fname) + 1;
                fprintf('  %-6s H%d: target=%.1f Hz, found=%.1f Hz, amp=%.4f\n', ...
                        fname, h, target, found_hz, max_amp);
            end
        end
    end
end

if isempty(detected_tones)
    fprintf('  No fault tones detected above threshold.\n');
end


%% ── 7. Fault Severity Score ──────────────────────────────────────────────────

% Kurtosis sub-score (0–100): normalise [1,10] → [100,0]
k_score = 100 * max(0, 1 - (kurt_val - 1) / 9);
k_score = max(0, min(100, k_score));

% Crest factor sub-score
cf_score = 100 * max(0, 1 - (cf_val - 1.4) / 10);
cf_score = max(0, min(100, cf_score));

% Tone count sub-score
total_tones = tone_count.BPFO + tone_count.BPFI + tone_count.BSF + tone_count.FTF;
tone_score  = 100 * max(0, 1 - total_tones / (n_harmonics * 4));

% Weighted health score
health_score = 0.30 * k_score + 0.25 * cf_score + 0.45 * tone_score;
health_score = max(0, min(100, health_score));

% Status
if health_score >= 80
    status = 'HEALTHY';    status_color = [0.15 0.68 0.38];
elseif health_score >= 60
    status = 'WARNING';    status_color = [0.95 0.61 0.07];
elseif health_score >= 40
    status = 'DEGRADED';   status_color = [0.90 0.50 0.13];
else
    status = 'FAULT';      status_color = [0.91 0.30 0.24];
end

fprintf('\n── Health Score ─────────────────────────────────────────────────\n');
fprintf('  Kurtosis sub-score     : %5.1f / 100  (kurtosis=%.2f)\n', k_score, kurt_val);
fprintf('  Crest factor sub-score : %5.1f / 100  (CF=%.2f)\n', cf_score, cf_val);
fprintf('  Tone count sub-score   : %5.1f / 100  (%d tones detected)\n', tone_score, total_tones);
fprintf('  ────────────────────────────────────────────\n');
fprintf('  ASSET HEALTH SCORE     : %5.1f / 100\n', health_score);
fprintf('  STATUS                 : %s\n\n', status);


%% ── 8. Plots ──────────────────────────────────────────────────────────────────

fig = figure('Name','Bearing Fault Detection','Position',[50 50 1400 900]);

% Panel A: Waveform (first 100 ms)
ax1 = subplot(3,2,1);
win = min(round(0.1*Fs), N);
plot(t(1:win)*1000, sig(1:win), 'Color',[0.16 0.50 0.73],'LineWidth',0.7);
xlabel('Time (ms)'); ylabel('Amplitude');
title(sprintf('Waveform — Kurt=%.2f  CF=%.2f', kurt_val, cf_val),'FontWeight','bold');
grid on;

% Panel B: FFT Spectrum
ax2 = subplot(3,2,2);
plot(f_axis, P1, 'Color',[0.16 0.16 0.16],'LineWidth',0.7);
hold on;
for fi = 1:numel(fault_hz)
    xline(fault_hz(fi),'--','Color',[0.8 0.3 0.3],'LineWidth',0.8);
end
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('FFT Spectrum','FontWeight','bold');
xlim([0 min(1000, Fs/2)]); grid on;

% Panel C: Envelope (time)
ax3 = subplot(3,2,3);
win_env = min(round(0.05*Fs), N);
plot(t(1:win_env)*1000, sig_bp(1:win_env),'Color',[0.6 0.6 0.6],'LineWidth',0.5); hold on;
plot(t(1:win_env)*1000, env(1:win_env),'Color',[0.90 0.30 0.24],'LineWidth',1.5);
xlabel('Time (ms)'); ylabel('Amplitude');
title('Band-pass Signal + Hilbert Envelope','FontWeight','bold');
legend({'Band-pass','Envelope'},'FontSize',8); grid on;

% Panel D: Envelope Spectrum
ax4 = subplot(3,2,4);
plot(f_axis, P_env, 'Color',[0.16 0.16 0.16],'LineWidth',0.7);
hold on;
fault_colors = {'r','b','g','m'};
fault_labels = {'BPFO','BPFI','BSF','FTF'};
for fi = 1:4
    for h = 1:n_harmonics
        xline(fault_hz(fi)*h,'--','Color',fault_colors{fi},'LineWidth',1.0,'Alpha',0.7);
    end
end
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('Envelope Spectrum','FontWeight','bold');
xlim([0 min(500, Fs/2)]); grid on;

% Add legend manually
hold on;
for fi = 1:4
    plot(nan, nan,'--','Color',fault_colors{fi},'LineWidth',1.5,'DisplayName',fault_labels{fi});
end
legend('show','FontSize',8,'Location','northeast');

% Panel E: Health Score Gauge
ax5 = subplot(3,2,5);
theta = linspace(pi, 0, 200);
r_outer = 1; r_inner = 0.6;
zones = [0 40 60 80 100];
zone_colors = {[0.91 0.30 0.24],[0.90 0.50 0.13],[0.95 0.61 0.07],[0.15 0.68 0.38]};
for zi = 1:4
    t1 = pi * (1 - zones(zi)/100);
    t2 = pi * (1 - zones(zi+1)/100);
    th = linspace(t1, t2, 50);
    px = [r_inner*cos(th), fliplr(r_outer*cos(th))];
    py = [r_inner*sin(th), fliplr(r_outer*sin(th))];
    fill(px, py, zone_colors{zi},'EdgeColor','white','LineWidth',0.5); hold on;
end
% Needle
needle_angle = pi * (1 - health_score/100);
plot([0, 0.85*cos(needle_angle)],[0, 0.85*sin(needle_angle)],'k-','LineWidth',3);
plot(0,0,'ko','MarkerSize',8,'MarkerFaceColor','k');
text(0, -0.2, sprintf('%.1f', health_score),'HorizontalAlignment','center',...
     'FontSize',18,'FontWeight','bold','Color',status_color);
text(0, -0.4, status,'HorizontalAlignment','center',...
     'FontSize',14,'FontWeight','bold','Color',status_color);
axis equal; axis off;
title('Asset Health Score','FontWeight','bold');

% Panel F: Tone Summary Table
ax6 = subplot(3,2,6);
axis off;
if ~isempty(detected_tones)
    col_names = {'Fault','H','Target(Hz)','Found(Hz)','Amp'};
    tbl_data  = cell(numel(detected_tones), 5);
    for i = 1:numel(detected_tones)
        tbl_data(i,:) = {detected_tones(i).name, detected_tones(i).harmonic, ...
            sprintf('%.1f',detected_tones(i).target_hz), ...
            sprintf('%.1f',detected_tones(i).found_hz), ...
            sprintf('%.4f',detected_tones(i).amplitude)};
    end
    t_obj = uitable(fig,'Data',tbl_data,'ColumnName',col_names,...
        'Units','normalized','Position',[0.55 0.05 0.42 0.28],...
        'FontSize',9,'RowName',[]);
else
    text(0.5, 0.5, 'No fault tones detected', 'HorizontalAlignment','center',...
         'FontSize',12, 'Color',[0.15 0.68 0.38],'FontWeight','bold');
end
title('Detected Fault Tones','FontWeight','bold');

sgtitle(sprintf('Bearing Fault Analysis  |  %.0f RPM  |  Health: %.1f/100  |  %s', ...
                shaft_rpm, health_score, status),...
        'FontSize',14,'FontWeight','bold');


%% ── 9. Export Results ────────────────────────────────────────────────────────

out_dir = fullfile('..', 'data', 'processed');
if ~isfolder(out_dir), mkdir(out_dir); end

results_tbl = table( ...
    rms_val, kurt_val, cf_val, pp_val, skew_val, ...
    total_tones, health_score, string(status), ...
    'VariableNames', {'rms','kurtosis','crest_factor','peak_to_peak','skewness', ...
                      'fault_tones_detected','health_score','status'});
out_csv = fullfile(out_dir, 'matlab_bearing_diagnosis.csv');
writetable(results_tbl, out_csv);
fprintf('[OK]   Results exported → %s\n', out_csv);

% Save figure
saveas(fig, fullfile(out_dir, 'matlab_bearing_analysis.png'));
fprintf('[OK]   Plot saved → matlab_bearing_analysis.png\n\n');
