clear all; close all; clc

%% Initializing Fourier Domain

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); 
y = x; 
z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);
[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%% Algorithm 1: Fast Fourier Transform + Signal Averaging for Center Frequency
load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata
total = zeros(n,n,n);
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    total = total + fftshift(fftn(Un));
end
avg = abs(total)/49;
[~, index] = max(avg(:));
[i1,j1,k1] = ind2sub(size(avg),index); 
k0_x = ks(j1); 
k0_y = ks(i1); 
k0_z = ks(k1);

%% Algorithm 2: Gaussian Filtering for Trajectory
tau = 0.05;
filter = exp(-tau*(((Kx-k0_x).^2)+((Ky-k0_y).^2)+((Kz-k0_z).^2)));
coordinates = zeros(49, 3);
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Un_fft = fftshift(fftn(Un));
    Un_filter = Un_fft .* filter;
    Un_res = ifftn(Un_filter);
    [~, index] = max(Un_res(:));
    [i2,j2,k2] = ind2sub(size(Un_res),index);
    coordinates(j,1) = X(i2,j2,k2); 
    coordinates(j,2) = Y(i2,j2,k2); 
    coordinates(j,3) = Z(i2,j2,k2);
end
hold on
plot3(coordinates(:,1),coordinates(:,2),coordinates(:,3),'-o','MarkerSize',3)
plot3(coordinates(1,1),coordinates(1,2),coordinates(1,3),'g.','MarkerSize',15)
plot3(coordinates(end,1),coordinates(end,2),coordinates(end,3),'r.','MarkerSize',15)
title('Submarine Trajectory')
legend('Submarine Path','Starting Location','Final Location')
xlabel('x-position'); ylabel('y-position'); zlabel('z-position')
view(45,20)
axis([-L L -L L -L L]), grid on, drawnow
