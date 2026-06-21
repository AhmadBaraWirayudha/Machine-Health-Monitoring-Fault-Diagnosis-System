%% Vibration Signal Analysis — MATLAB
% Machine Health Monitoring & Fault Diagnosis System
% ---------------------------------------------------------
% Performs: FFT, RMS, Kurtosis, Envelope Analysis
% Input:    CSV of raw vibration data (one column per bearing)
% Output:   Plots + feature table saved to ../data/processed/
%
% Usage:
%   Run section-by-section (Ctrl+Enter) or run the full script.
%   Ensure `vibration_data.csv` is in ../data/raw/

clear; clc; close all;

%% ── Parameters ──────────────────────────────────────────────────────────────
Fs          = 25600;        % Sampling rate [Hz]
signal_col  = 1;            % Which bearing column to analyse (1-indexed)
data_file   = fullfile('..', 'data', 'raw', 'vibration_data.csv');
output_dir  = fullfile('..', 'data', 'processed');

%% ── Load data ───────────────────────────────────────────────────────────────
if ~isfile(data_file)
    fprintf('[WARN] Data file not found: %s\n', data_file);
    fprintf('       Generating synthetic demo signal instead.\n');
    t   = (0 : 1/Fs : 1 - 1/Fs)';          % 1-second window
    sig = sin(2*pi*60*t) ...               % Shaft frequency
        + 0.3*sin(2*pi*120*t) ...          % Bearing fault frequency
        + 0.1*randn(size(t));              % Noise
    % Add impulsive bearing faults
    impulse_idx = randperm(length(sig), 20);
    sig(impulse_idx) = sig(impulse_idx) + 3;
else
    raw = readmatrix(data_file);
    sig = raw(:, signal_col);
    t   = (0 : length(sig)-1)' / Fs;
    fprintf('[OK]  Loaded %d samples from %s\n', length(sig), data_file);
end

N = length(sig);

%% ── 1. Time-domain Waveform ─────────────────────────────────────────────────
figure('Name', 'Vibration Waveform', 'Position', [100 600 900 280]);
plot(t*1000, sig, 'Color', [0.18 0.39 0.71], 'LineWidth', 0.6);
xlabel('Time (ms)'); ylabel('Amplitude');
title('Vibration Waveform', 'FontSize', 13, 'FontWeight', 'bold');
grid on; xlim([0 50]);

%% ── 2. FFT Spectrum ─────────────────────────────────────────────────────────
Y    = fft(sig);
P2   = abs(Y/N);
P1   = P2(1 : floor(N/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f    = Fs * (0 : floor(N/2)) / N;

[peak_amp, peak_idx] = max(P1);
peak_freq = f(peak_idx);

figure('Name', 'FFT Spectrum', 'Position', [100 250 900 280]);
plot(f, P1, 'Color', [0.13 0.13 0.13], 'LineWidth', 0.8);
hold on;
plot(peak_freq, peak_amp, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xline(peak_freq, '--r', sprintf('Peak: %.1f Hz', peak_freq), 'LabelHorizontalAlignment', 'left');
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('Single-Sided Amplitude Spectrum', 'FontSize', 13, 'FontWeight', 'bold');
grid on; xlim([0 Fs/2]);

%% ── 3. Envelope Analysis (demodulation) ─────────────────────────────────────
% Band-pass filter around bearing fault frequency, then envelope
bp_low  = 2000;     % Hz — adjust to bearing fault frequency band
bp_high = 8000;
[b_bp, a_bp] = butter(4, [bp_low, bp_high]/(Fs/2), 'bandpass');
sig_bp = filtfilt(b_bp, a_bp, sig);

envelope = abs(hilbert(sig_bp));     % Hilbert envelope

Y_env = fft(envelope);
P_env = abs(Y_env/N);
P_env = P_env(1 : floor(N/2)+1);
P_env(2:end-1) = 2*P_env(2:end-1);

figure('Name', 'Envelope Spectrum', 'Position', [100 0 900 280]);
subplot(2,1,1);
plot(t*1000, envelope, 'Color', [0.80 0.20 0.20], 'LineWidth', 0.7);
xlabel('Time (ms)'); ylabel('Envelope Amplitude');
title('Signal Envelope (Hilbert)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; xlim([0 50]);

subplot(2,1,2);
plot(f, P_env, 'Color', [0.13 0.13 0.13], 'LineWidth', 0.8);
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('Envelope Spectrum (BPFI/BPFO detection)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; xlim([0 500]);

%% ── 4. Time-domain feature calculation ──────────────────────────────────────
rms_val      = rms(sig);
kurtosis_val = kurtosis(sig);
crest_factor = max(abs(sig)) / rms_val;
peak_to_peak = max(sig) - min(sig);
skew_val     = skewness(sig);

fprintf('\n────────────────────────────────────\n');
fprintf('  Time-Domain Feature Summary\n');
fprintf('────────────────────────────────────\n');
fprintf('  RMS               : %8.4f\n', rms_val);
fprintf('  Kurtosis          : %8.4f\n', kurtosis_val);
fprintf('  Crest Factor      : %8.4f\n', crest_factor);
fprintf('  Peak-to-Peak      : %8.4f\n', peak_to_peak);
fprintf('  Skewness          : %8.4f\n', skew_val);
fprintf('  FFT Peak Freq     : %8.1f Hz\n', peak_freq);
fprintf('  FFT Peak Amp      : %8.4f\n', peak_amp);

%% ── 5. Health assessment (rule-based) ───────────────────────────────────────
fprintf('\n────────────────────────────────────\n');
fprintf('  Condition Assessment\n');
fprintf('────────────────────────────────────\n');

if kurtosis_val > 6 || crest_factor > 8
    status = 'CRITICAL — Bearing fault likely. Immediate inspection.';
elseif kurtosis_val > 4 || crest_factor > 5
    status = 'WARNING  — Elevated impulsiveness. Increase monitoring.';
else
    status = 'NORMAL   — Healthy operating signature.';
end
fprintf('  Status: %s\n\n', status);

%% ── 6. Export feature table to CSV ──────────────────────────────────────────
if ~isfolder(output_dir)
    mkdir(output_dir);
end

feature_table = table( ...
    rms_val, kurtosis_val, crest_factor, peak_to_peak, skew_val, ...
    peak_freq, peak_amp, ...
    'VariableNames', {'rms','kurtosis','crest_factor','peak_to_peak','skewness', ...
                      'fft_peak_freq','fft_peak_amplitude'} ...
);

out_csv = fullfile(output_dir, 'matlab_features.csv');
writetable(feature_table, out_csv);
fprintf('[OK]  Features exported → %s\n', out_csv);
