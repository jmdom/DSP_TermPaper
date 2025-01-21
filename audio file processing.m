clear, clc
[y, fs] = audioread('Audio\201101070538.wav');

% time domain of "Normal"
t = linspace(0,length(y)/fs,length(y));
figure;
plot(t,y);
title('temporal domain of Normal')
xlabel('time')
ylabel('amplitude')
length(y)

S1 = (0.001*sin(2*pi*500*t))';
S2 = (0.001*sin(2*pi*600*t))';
S3 = (0.001*sin(2*pi*700*t))';
NSignal = S1 + S2 + S3 + y;

% FFT parameters
NFFT = 12000;  % Number of FFT points 
magFrames = abs(fft(NSignal, NFFT));  % Compute magnitude of the FFT

% Frequency vector for plotting
f = linspace(0, NFFT/2, NFFT/2+1);  % Frequency axis from 0 to Nyquist frequency

% Find the index corresponding to 1600 Hz
maxFreq = 1600;  % Maximum frequency to plot (1600 Hz)
maxIdx = find(f <= maxFreq, 1, 'last');  % Find the index of the closest frequency <= 1600 Hz

% Plot energy spectrum up to 1600 Hz
figure;
plot(f(1:maxIdx), magFrames(1:maxIdx));  % Plot only up to 1600 Hz
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('magnitude Spectrum of the Signal (up to 1600 Hz)');
grid on;  % Add grid for better visualization

%% Add 3 sinusoidal noise signal

S1 = (0.1*sin(2*pi*100*t))'
S2 = (0.1*sin(2*pi*200*t))'
S3 = (0.1*sin(2*pi*300*t))'
NSignal = S1 + S2 + S3 + y
title('temporal domain of Noisy Signal')
xlabel('time')
ylabel('amplitude')
plot(t,NSignal)

%% Compute Mel Frequency Cepstral Coefficients
clear, clc
[y, fs] = audioread('Audio\adultCASE1.mp3');


aFE = audioFeatureExtractor(SampleRate=fs,melSpectrum=true,spectralRolloffPoint=true);
setExtractorParameters(aFE,"melSpectrum",NumBands=20)
setExtractorParameters(aFE,"spectralRolloffPoint",Threshold=0.8)
features = extract(aFE,y);
idx = info(aFE);
surf(10*log10(features(:,idx.melSpectrum)))
title("Mel Spectrum")


plot(features(:,idx.spectralRolloffPoint))
title("Spectral Rolloff Point")
xlabel('frequency')
ylabel('magnitude')

%% 
clear,clc
t=0:0.01:1
f=0
y= sin(2*pi*f*t)
plot(t,y)
hold on
f=1
y= sin(2*pi*f*t)
plot(t,y)
hold on
f=2
y= sin(2*pi*f*t)
plot(t,y)
hold on
f=3
y= sin(2*pi*f*t)
plot(t,y)
hold on
xlabel('time')
legend('k=0','k=1','k=2','k=3')
ylabel('amplitude')
%% 
clear,clc
% Clear workspace and command window
clear; clc;

% Load an audio signal
[audioIn, fs] = audioread('Audio\Atraining_normal\201101070538.wav');

% Parameters
% Set the parameters for CWT
scales = 1:128; % Define the range of scales
waveletName = 'morl'; % Morlet wavelet

% Step 1: Compute the Continuous Wavelet Transform (CWT)
% Compute CWT
[cwtCoefficients, frequencies] = cwt(audioIn,'morl',seconds(1/fs));

% Step 2: Plot the scalogram
figure;
imagesc(linspace(0, length(audioIn)/fs, size(cwtCoefficients, 2)), ...
         frequencies, ...
         abs(cwtCoefficients));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Scalogram (CWT)');
colormap(jet);
colorbar;
ylim([0 fs/2]); % Limit y-axis to Nyquist frequency
xlim([0 length(audioIn)/fs]); % Limit x-axis to signal duration


%% 
clear, clc
% audio signal
[audioIn, fs] = audioread('Audio\Atraining_normal\201104141251.wav');

% Pre-emphasize the signal 
preEmphasis = 0.97;
audioIn = filter([1 -preEmphasis], 1, audioIn);


% Frame the signal 
% audioIn = 156720 data points
% sampling frequency = 11025 data points/s
% audio duration = 14sec
frameSize = 0.025; 
frameStep = 0.010;   
frameLength = round(frameSize * fs);  % Length of each frame in samples
frameStep = round(frameStep * fs);    % Step between successive frames
frames = buffer(audioIn, frameLength, frameLength - frameStep, 'nodelay');

% Apply the Hamming window to each frame
hammingWindow = hamming(frameLength);
windowedFrames = frames .* hammingWindow;

%1424 frames from 14 sec  - audio file with 276 data points/ frame 

figure;
plot((0:frameLength-1)/fs, frames(:, 1), 'b', 'DisplayName', 'Original Frame');
hold on;
plot((0:frameLength-1)/fs, windowedFrames(:, 1), 'r', 'DisplayName', 'Hamming Windowed Frame');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original vs Hamming Windowed Signal (First Frame)');
legend;
grid on;


figure;
plot((0:frameLength-1)/fs, frames(:, 2), 'b', 'DisplayName', 'Original Frame');
hold on;
plot((0:frameLength-1)/fs, windowedFrames(:, 2), 'r', 'DisplayName', 'Hamming Windowed Frame');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original vs Hamming Windowed Signal (Second Frame)');
legend;
grid on;


%Take FFT and compute power spectrum
NFFT = floor(fs/2);  % Number of FFT points
magFrames = abs(fft(windowedFrames, NFFT)).^2;  % Magnitude spectrum
powerFrames = magFrames(1:NFFT/2+1, :);        % Power spectrum  nyquist


figure;
plot(linspace(0, NFFT/2, NFFT/2+1),powerFrames(:,1:10));
xlabel('Frequency (Hz)');
legend('frame1','frame2','frame3','frame4','frame5','frame6','frame7','frame8','frame9','frame10')
ylabel('Magnitude');
title('magniude of all windowedframes (before applying Mel filter bank)');
grid on;



figure;
plot(linspace(0, NFFT/2, NFFT/2+1), 10*log10(powerFrames(:,1:10))); 
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
legend('frame1','frame2','frame3','frame4','frame5','frame6','frame7','frame8','frame9','frame10')
title('magnitude spectrum of all windowedframes (in dB) (before applying Mel filter bank)');
grid on;

function melFilterBank = melFilterBank(numFilters, NFFT, fs)
    % melFilterBank - Generates a mel filter bank
    % numFilters    - Number of mel filters
    % NFFT          - Number of FFT points
    % fs            - Sampling frequency
    
    % Compute the low and high frequency limits in Hz
    lowFreq = 0;
    highFreq = fs / 2;
    
    % Convert frequency limits to Mel scale
    lowMel = 2595 * log10(1 + lowFreq / 700);
    highMel = 2595 * log10(1 + highFreq / 700);
   
    % Generate Mel filter bank points
    melPoints = linspace(lowMel, highMel, numFilters + 2);  % Add 2 for the edge filters
    hzPoints = 700 * (10.^(melPoints / 2595) - 1);  % Convert Mel scale back to Hz
    
    % Convert Hz to FFT bin numbers
    binPoints = floor((NFFT + 1) * hzPoints / fs);
   
    % Initialize filter bank
    melFilterBank = zeros(numFilters, NFFT / 2 + 1);
    
    % Create triangular filters
    for m = 2:(numFilters + 1)
        for k = binPoints(m-1):binPoints(m)
            melFilterBank(m-1, k+1) = (k - binPoints(m-1)) / (binPoints(m) - binPoints(m-1));
        end
        for k = binPoints(m):binPoints(m+1)
            melFilterBank(m-1, k+1) = (binPoints(m+1) - k) /  (binPoints(m+1) - binPoints(m));
        end
    end

    figure;
    plot(linspace(0, fs/2, NFFT/2+1), melFilterBank');
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    legend('Index1','Index2','Index3','Index4','Index5','Index6','Index7','Index8','Index9','Index10')
    title('Mel Filter Bank');
    grid on;
end

%Apply a Mel filter bank
numFilters = 26;  % Number of Mel filters
melFilter = melFilterBank(numFilters, NFFT, fs);  % Create a Mel filter bank

melEnergies = melFilter * powerFrames;            % Filter the power spectra

% Take the log of the Mel filter bank energies
logMelEnergies = log(melEnergies);  


figure;
plot(1:numFilters, logMelEnergies(:,1:10));
xlabel('Mel Filter Index');
ylabel('Log Energy');
legend('frame1','frame2','frame3','frame4','frame5','frame6','frame7','frame8','frame9','frame10')
title('Mel Energies of all windowedframes (after applying Mel filter bank)');
grid on;

% Take the DCT of the log Mel energies
numCoeffs = 20;
mfccs = dct(logMelEnergies);
mfccs = mfccs(1:numCoeffs, :);  % Keep the first 20 coefficients

% Transpose to have one row per frame and 20 columns (coefficients)
mfccs = mfccs';


% Plot the MFCCs
figure;
imagesc(mfccs');  
colormap(viridis(256)); 
xlabel('Frame');
ylabel('MFCC Coefficient');
title('Mel-Frequency Cepstral Coefficients (MFCCs)');
colorbar;
axis tight;



%% 
clear, clc
% Load an audio signal
[audioIn, fs] = audioread('Audio\Atraining_normal\201102270940.wav');
win = hann(1024,"periodic");
coeff=mfcc(audioIn,fs);

mfcc(audioIn,fs)

%% 
clear, clc
% Step 1: Read the audio file

[audioIn,fs] = audioread("Audio\adultCASE1.mp3");

% Compute the FFT (frequency domain)
N = length(audioIn);  % Number of samples
X = fft(audioIn);     % FFT of the signal
X_mag = abs(X(1:N/2+1));  % Magnitude spectrum (positive frequencies only)

% Convert the magnitude spectrum to dB scale
X_mag_dB = 20 * log10(X_mag + eps);  % Convert magnitude to dB (adding eps to avoid log(0))

%  Frequency vector
frequencies = (0:N/2) * fs / N;  % Frequency vector corresponding to the positive frequencies

% Plot the FFT in dB
figure;
plot(frequencies, X_mag_dB);  % Plot the FFT in dB
title('FFT of the Audio Signal (in dB)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;

%% 

clear, clc
[audioIn,fs] = audioread("Audio\adultCASE1.mp3");

centroid = spectralCentroid(audioIn,fs, ...
                            Window=hamming(fs), ...
                            OverlapLength=round(0.5*fs), ...
                            Range=[10,fs/2]);

spectralCentroid(audioIn,fs, ...
                 Window=hamming(fs), ...
                 OverlapLength=round(0.5*fs), ...
                 Range=[10,fs/2])

%% 
% Parameters
[audioIn, fs] = audioread('Audio\adultCASE1.mp3');  % Read audio file
windowLength = 11025;  % Length of each analysis window
overlap = round(windowLength / 2);  % 50% overlap

% Calculate the Short-Time Fourier Transform (STFT)
[S, F, T] = spectrogram(audioIn, windowLength, overlap, [], fs);  % Spectrogram

% Calculate the spectral centroid
spectralCentroid = sum(F .* abs(S), 1) ./ sum(abs(S), 1);  % Spectral centroid for each time frame

% Plotting
figure;
plot(T, spectralCentroid);  % Time vs Spectral Centroid
title('Spectral Centroid Over Time');
xlabel('Time (s)');
ylabel('Spectral Centroid (Hz)');
grid on;


%% Spectogram
clear,clc
[audioIn, fs] = audioread('Audio\Atraining_normal\201101070538.wav');  


windowLength = round(44100/4);  
overlap = 0; 
nfft = windowLength; 
[S, F, T] = spectrogram(audioIn, windowLength, overlap, nfft, fs); 



figure;
imagesc(T, F, 10*log10(abs(S)));  
axis xy; 
colorbar; 
title('Short-Time Fourier Transform (STFT)');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
grid on;



%% 
% Clear all variables and the command window
clear; clc;

% Introduction message
disp('--- Determinants and Span of Vectors ---');

% Define a 3x3 matrix (you can modify this to experiment with different matrices)
A = [1 2 3; 4 5 6; 7 8 10];

% Display the matrix A
disp('Matrix A:');
disp(A);

% Compute and display the determinant of matrix A
det_A = det(A);
disp(['Determinant of matrix A: ', num2str(det_A)]);

% Check if the determinant is zero
if det_A == 0
    disp('The columns of A are linearly dependent (they do NOT span R^n).');
else
    disp('The columns of A are linearly independent (they span R^n).');
end

% Illustrate the concept of span using a 2D example
disp('--- Span of two vectors in 2D ---');

% Define two 2D vectors
v1 = [1; 2];
v2 = [3; 4];

% Display the vectors
disp('Vector v1:');
disp(v1);
disp('Vector v2:');
disp(v2);

% Create a matrix V with vectors v1 and v2 as columns
V = [v1 v2];

% Compute and display the determinant of matrix V
det_V = det(V);
disp(['Determinant of V: ', num2str(det_V)]);

% Check if the determinant is zero
if det_V == 0
    disp('The vectors v1 and v2 are linearly dependent (they do NOT span R^2).');
else
    disp('The vectors v1 and v2 are linearly independent (they span R^2).');
end

% Visualization of the span in 2D
figure;
hold on;
grid on;

% Plot vector v1
quiver(0, 0, v1(1), v1(2), 'r', 'LineWidth', 2);
text(v1(1), v1(2), 'v1', 'FontSize', 12);

% Plot vector v2
quiver(0, 0, v2(1), v2(2), 'b', 'LineWidth', 2);
text(v2(1), v2(2), 'v2', 'FontSize', 12);

% Plot a few combinations 
%% Real part of the DFT Matrix (256 points)
clear,clc

w = exp(-i*2*pi/256);

for i=1:256
    for j = 1:256
        DFT(i,j) = w^((i-1)*(j-1));
    end
end

imagesc(real(DFT))
%% 
clear;
clc;

% Load audio file
[y, fs] = audioread('Audio/adultCASE1.mp3');

% Time domain of "Normal"
t = linspace(0, length(y)/fs, length(y));
figure;
plot(t, y);
title('Temporal Domain of Normal');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

% Generate noisy/unwanted tone signals to add to the original signal
S1 = (0.005 * sin(2 * pi * 800 * t(1:length(y))))';  % 800 Hz sine wave
S2 = (0.005* sin(2 * pi * 900 * t(1:length(y))))';  % 900 Hz sine wave
S3 = (0.005 * sin(2 * pi * 1000 * t(1:length(y))))';  % 1000 Hz sine wave

% Combine signals
NSignal = S1 + S2 + S3 + y;

% Plot the resulting signal in time domain
figure;
plot(t, NSignal)
title('Temporal Domain of Noisy Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

% FFT parameters
NFFT = 3200;  % Number of FFT points 
magFrames = abs(fft(NSignal, NFFT));  % Compute magnitude of the FFT

% Frequency vector for plotting
f = linspace(0, fs/2, NFFT/2+1);  % Frequency axis from 0 to Nyquist frequency

% Find the index corresponding to 1600 Hz
maxFreq = 1600;  % Maximum frequency to plot (1600 Hz)
maxIdx = find(f <= maxFreq, 1, 'last');  % Find the index of the closest frequency <= 1600 Hz

% Plot magnitude spectrum up to 1600 Hz
figure;
plot(f(1:maxIdx), magFrames(1:maxIdx));  % Plot only up to 1600 Hz
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('magnitude spectrum of the Signal (up to 1600 Hz)');
grid on;  

%% Fourier series approximation to a hat function
clear,clc
%define domain
dx = 0.001;
L = pi;
x = (-1+0.001:dx:1) * L;
n = length(x); nquart = floor(n/4);

%define hat function
f  = 0 * x;
f(nquart:2*nquart) = 4*(1:nquart+1)/n;
f(2*nquart+1:3*nquart) = 1 -4*(0:nquart-1)/n;
plot(x,f,'-k','LineWidth',1.5), hold on

%Compute Fourier series
CC = jet(20);
A0 = sum(f.*ones(size(x))*dx);
fFS = A0/2;
for k=1:20
    A(k) = sum(f.*cos(pi*k*x/L))*dx;
    B(k) = sum(f.*sin(pi*k*x/L))*dx;
    fFS = fFS + A(k)*cos(k*pi*x/L) + B(k)*sin(k*pi*x/L);
    plot(x,fFS,'-','Color',CC(k,:),'LineWidth',1.2)
end


%% APPLICATION OF SVD
clear,clc
A = imread('picture.jpg');
X = double(rgb2gray(A)); %256bit -> double
nx = size(X,1);
ny = size(X,2);

imagesc(X), axis off, colormap gray

%SVD
[U,S,V] = svd(X);

%Approximate matric with truncated SVD for various ranks r
for r=[5 10 20 50 300]
    Xapprox = U(:,1:r)* S(1:r,1:r)*V(:,1:r)';
    figure, imagesc(Xapprox), axis off
    title(['r=',num2str(r,'%d')]);
end

%% 
clear,clc
% Fixed image dimensions
image_width = 100;
image_height = 100;
fixed_size = image_width * image_height;  % Total number of pixels in the image
%[audio, fs] = audioread('Audio/adultCASE1.mp3');
% Step 1: Load the Audio File
 [audio, fs] = audioread('Audio\Atraining_normal\201101070538.wav');  % Replace with your audio file name

% Step 2: Resample the Audio Signal to Fit the Fixed Size
audio_resampled = resample(audio, fixed_size, length(audio));

% Step 3: Normalize the Signal to the Range [0, 255]
audio_normalized = (audio_resampled - min(audio_resampled)) / (max(audio_resampled) - min(audio_resampled)) * 255;

% Step 4: Convert to Unsigned 8-bit Integer (0 to 255 grayscale range)
audio_uint8 = uint8(audio_normalized);

% Step 5: Reshape the Audio Signal into a Fixed 256x256 Image
audio_image = reshape(audio_uint8, image_width, image_height);

% Step 6: Display and Save the Image
imshow(audio_image);                      % Display the grayscale image


%% 
clear;
clc;

% Fixed image dimensions
image_width = 1000;
image_height = 1000;
fixed_size = image_width * image_height;  % Total number of pixels in the image

% Step 1: Load the Audio File
[audio, fs] = audioread('Audio\Atraining_normal\201101070538.wav');  % Replace with your audio file name

% Step 2: Convert to Mono if Stereo
if size(audio, 2) > 1
    audio = mean(audio, 2);  % Average the channels to get mono
end

% Step 3: Resample/Interpolate the Audio Signal to Fit the Fixed Size
original_length = length(audio);
x_original = linspace(1, original_length, original_length);
x_fixed = linspace(1, original_length, fixed_size);
audio_resampled = interp1(x_original, audio, x_fixed);

% Step 4: Normalize the Signal to the Range [0, 255]
audio_normalized = (audio_resampled - min(audio_resampled)) / (max(audio_resampled) - min(audio_resampled)) * 255;

% Step 5: Convert to Unsigned 8-bit Integer (0 to 255 grayscale range)
audio_uint8 = uint8(audio_normalized);

% Step 6: Reshape the Audio Signal into a Fixed 1080x2040 Image
audio_image = reshape(audio_uint8, image_width, image_height);

% Step 7: Display and Save the Image
imshow(audio_image);                       % Display the grayscale image
imwrite(audio_image, 'audio_fixed_size_grayscale.png');  % Save the image as a PNG

% Plot the original and interpolated signals
figure;
subplot(2, 1, 1);
plot(x_original, audio);
title('Original Audio Signal');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(x_fixed, audio_resampled);
title('Interpolated Audio Signal');
xlabel('Sample Number');
ylabel('Amplitude');


%% 
%% 
clear;
clc;


image_width = 256;
image_height = 256;
fixed_size = image_width * image_height;  

[audio, fs] = audioread('Audio\Atraining_normal\201101070538.wav');  


original_length = length(audio);
x_original = linspace(1, original_length, original_length);
x_fixed = linspace(1, original_length, fixed_size);


audio_resampled = interp1(x_original, audio, x_fixed);


audio_normalized = mat2gray(audio_resampled);  


audio_image = reshape(audio_normalized, image_width, image_height);


imagesc(audio_image);                                      
colormap(viridis(256));  % Apply the Parula colormap (MATLAB's default)
colorbar;


% Convert the image to RGB using Parula colormap and save it
rgb_image = ind2rgb(uint8(audio_image * 255), viridis(256));  % Convert to RGB format
imwrite(rgb_image, 'audio_fixed_size_parula.png');  % Save as PNG
%% 
% Wavelet Spectrogram with Morlet in MATLAB
clc; clear; close all;

[audio, fs] = audioread('201106111136.wav');  
 cwt(audio,"amor",fs);
 fig
 colormap viridis;

%% 
% Set the input and output folder paths
clc; clear; close all;
input_folder = '\Users\ROG\Documents\MATLAB\Audio\Atraining_artifact\'; 


% Get a list of all audio files in the input folder
audio_files = dir(fullfile(input_folder, '*.wav')); 

% Loop through each audio file
for i = 1:length(audio_files)
    % Load the audio file
    [audio, fs] = audioread(fullfile(input_folder, audio_files(i).name)); 

    % Calculate wavelet spectrogram using 'amor'
    [wt,f] = cwt(audio, 'amor', fs); 

    % Create the output filename
    output_filename = [audio_files(i).name(1:end-4), '_wavelet_spectrogram.png']; 


    % Plot the spectrogram
   
    figure('Visible', 'off');
    imagesc(abs(wt)); 
    colormap viridis;
    colorbar('off')
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    xlabel('')
    ylabel('')
    set(gca,'position',[0,0,1,1])

    % Save the spectrogram as an image
    saveas(gcf, output_filename); 
   
    % Close the figure to avoid memory issues
    close(gcf); 
end

disp('Spectrogram generation completed.');