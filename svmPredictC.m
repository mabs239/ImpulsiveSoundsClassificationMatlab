%%
function main
    close all;
    clear all;
    clc;
    % fet = [1 2]; % Select two features from 6
    % calculateFeatures();
%     load('features.mat');
    [X y] = getBinaryFlaggedFeatures();

%     x1 = [1 2 1]; 
%     x2 = [1 -4 0]; 
%     x1=1;x2=1;
%     sigma = 5;
%     C = 20; 
%     sigma = 0.1;

%     model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma))
%     save('model.mat', 'model');
    load('model.mat')
    
%     visualizeBoundary(X, y, model);

%     XSample=[.7 .7;2 0.6; .7 .4];
    XSample=[.7 .7]
    svmPredict(model, XSample)
    
%     x1=1; x2=0; sigma = .01;    gk = gaussianKernel(x1, x2, sigma)
    
    fprintf('   ...Program End...\n');
    
%     x=3
%     sqr = @(x,y) x.^y
%     a = sqr(5,2)
%     add(3,4)
    
  
    
end
%%
function y=add(a,b,z)

    y=a+b;
    
end

%%
% ========================================================
%   Assign Binary Flag To Features Vector
%   
% ========================================================
function [xdata group] = getBinaryFlaggedFeatures()
    calculateFeatures();
    load('features.mat');
    fet = [1 2]; % Select Features
    
    % Features{1} =  Features{1}(randperm(end),:);
    % Features{2} =  Features{2}(randperm(end),:);
    % Features{3} =  Features{3}(randperm(end),:);
        
    Features{1} = Features{1}(1:50,fet); % Shorten features
    Features{2} = Features{2}(:,fet); % Shorten features
    Features{3} = Features{3}(:,fet); % Shorten features
        
    lf1 = length(Features{1}) % yes
    lf2 = length(Features{2}) % no
    lf3 = length(Features{3}) % test
 
    xdata = [Features{1} ; Features{2}; Features{3}];
    % group = [repmat('''yes''',a,1); repmat('''no ''',b,1) ];
    group = [repmat(1,lf1,1); repmat(0,lf2,1); repmat(0,lf3,1)];
    
    
%     xdata = [Features{1}];
%     group = [repmat(1,lf1,1)];
end

%%
% ========================================================
%   Calculate Features with the directory names given
%   Uses computeAllStatistics(fileName, win, step)
% ========================================================
function calculateFeatures
    pathN = ['C:\Users\AbuBakar\Dropbox\' ...
                'SharedWithAbdulRasheed\Sounds\'];
%     classN = ({'yes\', 'no\','test\'});
    classN = ({'GunShot\', 'Balloon\','Clapping\'});
    
    % This function computes the audio features (6-D vector) for each .wav
    % file in all directories (given in classNames)
    Dim = 6; % Number of features per window
    win = 950; 
    step = 2600;
    FeaturesNames = {'Std Energy Entropy','Std/mean ZCR',... 
                    'Std Rolloff','Std Spectral Centroid', ...
                    'Std Spectral Flux','Std/mean Energy'};
    for c = 1:length(classN) % for each class (respective directories):
        fprintf('Computing features for class %s...\n',classN{c});
        [pathN '//' classN{c} '//*.wav']
        D = dir([pathN '//' classN{c} '//*.wav'])
        tempF = [];
        
        % tempF = zeros(length(D),Dim);
        for i = 1:length(D) % for each .wav file in the current directory:
            F = computeAllStatistics([pathN '//' classN{c} '//' D(i).name], win, step);     
            tempF = [tempF; F];
        end
        
        % keep a different cell element for each feature matrix:
        Features{c} = tempF;
    end
    
    Features % Prints dimension of the feature classes
    save('features.mat', 'Features');
end
%%
% ========================================================
%   Compute Statistics EE, ZCR, RollOff, Centroid, Flux
%   with given file name, window length and step
% ========================================================
function FF = computeAllStatistics(fileName, win, step)
    % This function computes the average and std values for the following audio
    % features:
    % EE - energy entropy
    % Z - zero crossing rate
    % E - short time energy
    % R - spectral rolloff
    % C - spectral centroid
    % F - spectral flux
    [x, fs] = wavread(fileName);
    EE = Energy_Entropy_Block(x, win, step, 10);
    Z = zcr(x, win, step, fs);
    R = SpectralRollOff(x, win, step, 0.80, fs);
    C = SpectralCentroid(x, win, step, fs);
    F = SpectralFlux(x, win, step, fs);
    E = ShortTimeEnergy(x, win, step);
    FF = [EE' Z R' C F E];
end
%%
%======================================================%
% SVM TRAIN %
%======================================================%

function [model] = svmTrain(X, Y, C, kernelFunction, ...
                            tol, max_passes)
                        
%SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
%algorithm. 
%   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
%   SVM classifier and returns trained model. X is the matrix of training 
%   examples.  Each row is a training example, and the jth column holds the 
%   jth feature.  Y is a column matrix containing 1 for positive examples 
%   and 0 for negative examples.  C is the standard SVM regularization 
%   parameter.  tol is a tolerance value used for determining equality of 
%   floating point numbers. max_passes controls the number of iterations
%   over the dataset (without changes to alpha) before the algorithm quits.
%
% Note: This is a simplified version of the SMO algorithm for training
%       SVMs. In practice, if you want to train an SVM classifier, we
%       recommend using an optimized package such as:  
%
%           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
%           SVMLight (http://svmlight.joachims.org/)
%
%
sprintf('Entering svmTrain ...')

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
Y(Y==0) = -1;

% Variables
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

% Pre-compute the Kernel Matrix since our dataset is small
% (in practice, optimized SVM packages that handle large datasets
%  gracefully will _not_ do this)
% 
% We have implemented optimized vectorized version of the Kernels here so
% that the svm training will run faster.
if strcmp(func2str(kernelFunction), 'linearKernel')
    % Vectorized computation for the Linear Kernel
    % This is equivalent to computing the kernel on every pair of examples
    K = X*X';
elseif strfind(func2str(kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = kernelFunction(1, 0) .^ K
else
    % Pre-compute the Kernel Matrix
    % The following can be slow due to the lack of vectorization
    K = zeros(m);
    for i = 1:m
        for j = i:m
             K(i,j) = kernelFunction(X(i,:)', X(j,:)');
             K(j,i) = K(i,j); %the matrix is symmetric
        end
    end
end

% Train
fprintf('\nTraining ...');
dots = 12;
while passes < max_passes,
            
    num_changed_alphas = 0;
    for i = 1:m,
        
        % Calculate Ei = f(x(i)) - y(i) using (2). 
        % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
        E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
        
        if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0)),
            
            % In practice, there are many heuristics one can use to select
            % the i and j. In this simplified code, we select them randomly.
            j = ceil(m * rand());
            while j == i,  % Make sure i \neq j
                j = ceil(m * rand());
            end

            % Calculate Ej = f(x(j)) - y(j) using (2).
            E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % Compute L and H by (10) or (11). 
            if (Y(i) == Y(j)),
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H),
                % continue to next i. 
                continue;
            end

            % Compute eta by (14).
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0),
                % continue to next i. 
                continue;
            end
            
            % Compute and clip new value for alpha j using (12) and (15).
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            
            % Clip
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            % Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol),
                % continue to next i. 
                % replace anyway
                alphas(j) = alpha_j_old;
                continue;
            end
            
            % Determine value for alpha i using (16). 
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            % Compute b1 and b2 using (17) and (18) respectively. 
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

            % Compute b by (19). 
            if (0 < alphas(i) && alphas(i) < C),
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C),
                b = b2;
            else
                b = (b1+b2)/2;
            end

            num_changed_alphas = num_changed_alphas + 1;

        end
        
    end
    
    if (num_changed_alphas == 0),
        passes = passes + 1;
    else
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end
fprintf(' Done! \n\n');

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';

end
%%
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%    
sim = exp(-((x1 - x2)'*(x1 - x2))/(2 * sigma^2));
[x1 x2 sigma sim sim^.01]
% =============================================================
    
end
%%
function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;

end
%%
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
%
% Note: This was slightly modified such that it expects y = 1 or y = 0

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold off;

end
%%

function pred = svmPredict(model, X)

%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain). 
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
%   trained SVM model (svmTrain). X is a mxn matrix where there each 
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end

% Dataset 
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

if strcmp(func2str(model.kernelFunction), 'linearKernel')
    % We can use the weights and bias directly if working with the 
    % linear kernel
    p = X * model.w + model.b;
elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    
    X=X
    model.X
    X1 = sum(X.^2, 2)
    X2 = sum(model.X.^2, 2)'
    K1 = - 2 * X * model.X'
    K2 = bsxfun(@plus, X2, K1)
    K3 = bsxfun(@plus, X1, K2)
    K4 = model.kernelFunction(1, 0) .^ K3
    
    model.y'
    K5 = bsxfun(@times, model.y', K4)
    model.alphas'
    
    K6 = bsxfun(@times, model.alphas', K5)
    p = sum(K6, 2)
else
    % Other Non-linear kernel
    for i = 1:m
        prediction = 0;
        for j = 1:size(model.X, 1)
            prediction = prediction + ...
                model.alphas(j) * model.y(j) * ...
                model.kernelFunction(X(i,:)', model.X(j,:)');
        end
        p(i) = prediction + model.b;
    end
end

% Convert predictions into 0 / 1
p
pred(p >= 0) =  1;
pred(p <  0) =  0;

end
%%

function [Entropy] = Energy_Entropy_Block(f,winLength,winStep,numOfShortBlocks)

f = f / max(abs(f));
Eol = sum(f.^2);
L = length(f);

if (winLength==0)
    winLength = floor(L);
    winStep = floor(L);
end


numOfBlocks = (L-winLength)/winStep + 1;
curPos = 1;
for (i=1:numOfBlocks)
    curBlock = f(curPos:curPos+winLength-1);
    for (j=1:numOfShortBlocks)        
        s(j) = sum(curBlock((j-1)*(winLength/numOfShortBlocks)+1:j*(winLength/numOfShortBlocks)).^2)/Eol;
    end
    
    Entropy(i) = -sum(s.*log2(s));
    curPos = curPos + winStep;
end

end




%%
function Z = zcr(signal,windowLength, step, fs);
signal = signal / max(abs(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
%H = hamming(windowLength);
Z = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));    
    window2 = zeros(size(window));
    window2(2:end) = window(1:end-1);
    Z(i) = (1/(2*windowLength)) * sum(abs(sign(window)-sign(window2)));
    curPos = curPos + step;
end

end


%%
function mC = SpectralRollOff(signal,windowLength, step, c, fs)
signal = signal / max(abs(signal));
curPos = 1;
L = length(signal);
numOfFrames = (L-windowLength)/step + 1;
H = hamming(windowLength);
m = [0:windowLength-1]';
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));    
    FFT = (abs(fft(window,512)));
    FFT = FFT(1:255);
    totalEnergy = sum(FFT);
    curEnergy = 0.0;
    countFFT = 1;
    while ((curEnergy<=c*totalEnergy) && (countFFT<=255))
        curEnergy = curEnergy + FFT(countFFT);
        countFFT = countFFT + 1;
    end
    mC(i) = ((countFFT-1))/(fs/2);
    curPos = curPos + step;
end
end

%%
function En = SpectralEntropy(signal,windowLength,windowStep, fftLength, numOfBins);
signal = signal / max(abs(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/windowStep) + 1;
H = hamming(windowLength);
En = zeros(numOfFrames,1);
h_step = fftLength / numOfBins;

for (i=1:numOfFrames)
    window = (H.*signal(curPos:curPos+windowLength-1));
    fftTemp = abs(fft(window,2*fftLength));
    fftTemp = fftTemp(1:fftLength);
    S = sum(fftTemp);    
    
    for (j=1:numOfBins)
        x(j) = sum(fftTemp((j-1)*h_step + 1: j*h_step)) / S;
    end
    En(i) = -sum(x.*log2(x));
    curPos = curPos + windowStep;
end
end



%%
function C = SpectralCentroid(signal,windowLength, step, fs)
signal = signal / max(abs(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
H = hamming(windowLength);
m = ((fs/(2*windowLength))*[1:windowLength])';
C = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = H.*(signal(curPos:curPos+windowLength-1));    
    FFT = (abs(fft(window,2*windowLength)));
    FFT = FFT(1:windowLength);  
    FFT = FFT / max(FFT);
    C(i) = sum(m.*FFT)/sum(FFT);
    if (sum(window.^2)<0.010)
        C(i) = 0.0;
    end
    curPos = curPos + step;
end
C = C / (fs/2);
end


%%
function F = SpectralFlux(signal,windowLength, step, fs)
signal = signal / max(abs(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
H = hamming(windowLength);
m = [0:windowLength-1]';
F = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = H.*(signal(curPos:curPos+windowLength-1));    
    FFT = (abs(fft(window,2*windowLength)));
    FFT = FFT(1:windowLength);        
    FFT = FFT / max(FFT);
    if (i>1)
        F(i) = sum((FFT-FFTprev).^2);
    else
        F(i) = 0;
    end
    curPos = curPos + step;
    FFTprev = FFT;
end
end


%%
function E = ShortTimeEnergy(signal, windowLength,step);
signal = signal / max(max(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
%H = hamming(windowLength);
E = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));
    E(i) = (1/(windowLength)) * sum(abs(window.^2));
    curPos = curPos + step;
end
%Max = max(E);
%med = median(E);
%m = mean(E);
%a = length(find(E>2*med))/numOfFrames;
%b = length(find(E<(med/2)))/numOfFrames;

%S_EnergyM = length(find(E>2*med))/numOfFrames;
%S_EnergyV = length(find(E<(med/2)))/numOfFrames;
%S_EnergyM = std(E);
%S_EnergyV = a;
end