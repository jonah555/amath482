%% Read Ski Drop video

video = VideoReader('ski_drop_low.mp4');
t = linspace(0,video.duration,video.NumFrames);
dt = video.duration / video.NumFrames;
frames = read(video);
for j = 1:video.NumFrames
    frame = rgb2gray(frames(:,:,:,j)); 
    X(:,j) = frame(:);
end


%% DMD

X = im2double(X);
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X1,'econ');
S = U'*X2*V*diag(1./diag(Sigma));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors 
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV;

y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions 
u_modes = zeros(length(y0),video.NumFrames);
for iter = 1: video.NumFrames
    u_modes(:,iter) = y0.*exp(omega*t(iter));
end
u_dmd = Phi*u_modes;


%% Foreground and Background

sparse = X-abs(u_dmd(:,1));
foreground = mat2gray(reshape(sparse(:,200), [video.height video.width]));
figure()
imshow(foreground);
title('Foreground of Ski Drop at 200th frame')
saveas(gcf, 'ski_drop_foreground.png')

background = reshape(u_dmd(:,1), [video.height video.width]); 
figure()
imshow(background);
title('Background of Ski Drop')
saveas(gcf, 'ski_drop_background.png')



%% Read Monte Carlo video

clear;

video = VideoReader('monte_carlo_low.mp4'); 
t = linspace(0,video.duration,video.NumFrames);
dt = video.duration / video.NumFrames;
frames = read(video);
for j = 1:video.NumFrames
    frame = rgb2gray(frames(:,:,:,j)); 
    X(:,j) = frame(:);
end


%% DMD

X = im2double(X);
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X1,'econ');
S = U'*X2*V*diag(1./diag(Sigma));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors 
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV;


y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions 
u_modes = zeros(length(y0),video.NumFrames);
for iter = 1: video.NumFrames
    u_modes(:,iter) = y0.*exp(omega*t(iter));
end
u_dmd = Phi*u_modes;


%% Foreground and Background

sparse = X-abs(u_dmd(:,1));
foreground = mat2gray(reshape(sparse(:,200), [video.height video.width]));
figure()
imshow(foreground);
title('Foreground of Monte Carlo at 200th frame')
saveas(gcf, 'monte_carlo_foreground.png')

background = reshape(u_dmd(:,1), [video.height video.width]); 
figure()
imshow(background);
title('Background of Monte Carlo')
saveas(gcf, 'monte_carlo_background.png')
