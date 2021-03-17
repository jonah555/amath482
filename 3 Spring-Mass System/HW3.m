clear; close all; clc;

%% Ideal Case

load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')

% 1_1
ideal_1 = size(vidFrames1_1, 4);
ideal_x1 = zeros(1,ideal_1);
ideal_y1 = zeros(1,ideal_1);
for j = 1:ideal_1
    V = vidFrames1_1(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,ideal_x1(j)] = max(mean(max(V,[],1),3));
    [~,ideal_y1(j)] = max(mean(max(V,[],2),3));
end

% 2_1
ideal_2 = size(vidFrames2_1, 4);
ideal_x2 = zeros(1,ideal_2);
ideal_y2 = zeros(1,ideal_2);
for j = 1:ideal_2
    V = vidFrames2_1(:,:,:,j);
    V(1:80,:,:) = 0;
    V(400:480,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,380:640,:) = 0;
    [~,ideal_x2(j)] = max(sum(max(V,[],1),3));
    [~,ideal_y2(j)] = max(sum(max(V,[],2),3));
end

% 3_1
ideal_3 = size(vidFrames3_1, 4);
ideal_x3 = zeros(1,ideal_3);
ideal_y3 = zeros(1,ideal_3);
for j = 1:ideal_3
    V = vidFrames3_1(:,:,:,j);
    V(1:200,:,:) = 0;
    V(400:480,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,540:640,:) = 0;
    [~,ideal_x3(j)] = max(sum(max(V,[],1),3));
    [~,ideal_y3(j)] = max(sum(max(V,[],2),3));
end

% pca
X = [ideal_x1;ideal_y1;
     ideal_x2(10:235);ideal_y2(10:235);
     ideal_x3(1:226);ideal_y3(1:226)];
[~,n] = size(X);
X = X - repmat(mean(X,2),1,n);
[U,S,V] = svd(X, 'econ');
Y = U'*X;
sigs = diag(S).^2;
energy = sigs/sum(sigs);

figure()
plot(1:6, energy, 'k.', 'MarkerSize',20);
title("Ideal Case Energies");
xlabel("Dimension"); 
ylabel("Energy (%)");
saveas(gcf,'ideal_energy.png');

figure()
plot(1:226, Y(1,:));
title("Ideal Case PCA");
xlabel("Time");
ylabel("Displacement");  
legend("PC");
saveas(gcf,'ideal.png');

%% Noisy Case

load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')

% 1_2
noisy_1 = size(vidFrames1_2, 4);
noisy_x1 = zeros(1,noisy_1);
noisy_y1 = zeros(1,noisy_1);
for j = 1:noisy_1
    V = vidFrames1_2(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    
    [~,noisy_x1(j)] = max(mean(max(V,[],1),3));
    [~,noisy_y1(j)] = max(mean(max(V,[],2),3));
end

% 2_2
noisy_2 = size(vidFrames2_2, 4);
noisy_x2 = zeros(1,noisy_2);
noisy_y2 = zeros(1,noisy_2);
for j = 1:noisy_2
    V = vidFrames2_2(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    
    [~,noisy_x2(j)] = max(mean(max(V,[],1),3));
    [~,noisy_y2(j)] = max(mean(max(V,[],2),3));
end

% 3_2
noisy_3 = size(vidFrames3_2, 4);
noisy_x3 = zeros(1,noisy_3);
noisy_y3 = zeros(1,noisy_3);
for j = 1:noisy_3
    V = vidFrames3_2(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    
    [~,noisy_x3(j)] = max(mean(max(V,[],1),3));
    [~,noisy_y3(j)] = max(mean(max(V,[],2),3));
end

% PCA
X = [noisy_x1;noisy_y1;
     noisy_x2(20:333);noisy_y2(20:333);
     noisy_x3(1:314);noisy_y3(1:314)];
[~,n] = size(X);
X = X - repmat(mean(X,2),1,n);
[U,S,V] = svd(X, 'econ');
Y = U'*X;
sigs = diag(S).^2;
energy = sigs/sum(sigs);

figure()
plot(1:6, energy, 'k.', 'MarkerSize',20);
title("Noisy Case Energies");
xlabel("Dimension"); 
ylabel("Energy (%)");
saveas(gcf,'noisy_energy.png');

figure()
plot(1:314, Y(1,:),1:314, Y(2,:), 1:314, Y(3,:));
title("Noisy Case PCA");
xlabel("Time"); 
ylabel("Displacement"); 
legend("PC1", "PC2", "PC3");
saveas(gcf,'noisy.png');


%% Horizontal Displacement

load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

% 1_3
horizontal_1 = size(vidFrames1_3, 4);
horizontal_x1 = zeros(1,horizontal_1);
horizontal_y1 = zeros(1,horizontal_1);
for j = 1:horizontal_1
    V = vidFrames1_3(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,horizontal_x1(j)] = max(mean(max(V,[],1),3));
    [~,horizontal_y1(j)] = max(mean(max(V,[],2),3));
end

% 2_3
horizontal_2 = size(vidFrames2_3, 4);
horizontal_x2 = zeros(1,horizontal_2);
horizontal_y2 = zeros(1,horizontal_2);
for j = 1:horizontal_2
    V = vidFrames2_3(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,horizontal_x2(j)] = max(mean(max(V,[],1),3));
    [~,horizontal_y2(j)] = max(mean(max(V,[],2),3));
end

% 3_3
horizontal_3 = size(vidFrames3_3, 4);
horizontal_x3 = zeros(1,horizontal_3);
horizontal_y3 = zeros(1,horizontal_3);
for j = 1:horizontal_3
    V = vidFrames3_3(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,horizontal_x3(j)] = max(mean(max(V,[],1),3));
    [~,horizontal_y3(j)] = max(mean(max(V,[],2),3));
end

% pca
X = [horizontal_x1(1:237);horizontal_y1(1:237);
     horizontal_x2(35:271);horizontal_y2(35:271);
     horizontal_x3;horizontal_y3];
[~,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);
[U,S,V] = svd(X, 'econ');
Y = U'*X;
sigs = diag(S).^2;
energy = sigs/sum(sigs);

figure()
plot(1:6, energy, 'k.', 'MarkerSize',20);
title("Horizontal Displacement Energies")
xlabel("Dimension"); 
ylabel("Energy (%)");
saveas(gcf,'horizontal_energy.png')

figure()
plot(1:237, Y(1,:),1:237, Y(2,:), 1:237, Y(3,:))
title("Horizontal Displacement PCA");
xlabel("Time"); 
ylabel("Displacement"); 
legend("PC1", "PC2", "PC3")
saveas(gcf,'horizontal.png')


%% Horizontal Displacement and Rotation

load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')

% 1_4
rotatation_1 = size(vidFrames1_4, 4);
rotatation_x1 = zeros(1,rotatation_1);
rotatation_y1 = zeros(1,rotatation_1);
for j = 1:rotatation_1
    V = vidFrames1_4(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,rotatation_x1(j)] = max(mean(max(V,[],1),3));
    [~,rotatation_y1(j)] = max(mean(max(V,[],2),3));
end

% 2_4
rotatation_2 = size(vidFrames2_4, 4);
rotatation_x2 = zeros(1,rotatation_2);
rotatation_y2 = zeros(1,rotatation_2);
for j = 1:rotatation_2
    V = vidFrames2_4(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,rotatation_x2(j)] = max(mean(max(V,[],1),3));
    [~,rotatation_y2(j)] = max(mean(max(V,[],2),3));
end

% 3_4
rotate_3 = size(vidFrames3_4, 4);
rotate_x3 = zeros(1,rotate_3);
rotate_y3 = zeros(1,rotate_3);
for j = 1:rotate_3
    V = vidFrames3_4(:,:,:,j);
    V(1:180,:,:) = 0;
    V(:,1:240,:) = 0;
    V(:,460:640,:) = 0;
    [~,rotate_x3(j)] = max(mean(max(V,[],1),3));
    [~,rotate_y3(j)] = max(mean(max(V,[],2),3));
end

% pca
X = [rotatation_x1;rotatation_y1;
     rotatation_x2(14:405);rotatation_y2(14:405);
      rotate_x3(1:392);rotate_y3(1:392)];
[~,n] = size(X);
X = X - repmat(mean(X,2),1,n);
[U,S,V] = svd(X, 'econ');
Y = U'*X;
sigs = diag(S).^2;
energy = sigs/sum(sigs);

figure()
plot(1:6, energy, 'k.', 'MarkerSize',20);
title("Horizontal Displacement and Rotation Energies")
xlabel("Dimension"); 
ylabel("Energy (%)");
saveas(gcf,'rotation_energy.png')

figure()
plot(1:392, Y(1,:),1:392, Y(2,:),1:392, Y(3,:))
title("Horizontal Displacement and Rotation PCA");
xlabel("Time"); 
ylabel("Displacement"); 
legend("PC1", "PC2", "PC3")
saveas(gcf,'rotation.png')
