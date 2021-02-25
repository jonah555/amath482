%% GNR
[y1, Fs] = audioread('GNR.m4a');
tr_gnr = length(y1)/Fs;
n = length(y1);
L = tr_gnr;
k = (1/L)*[0:(n/2-1) -n/2:-1];
ks = fftshift(k);
a = 50;
ts = linspace(0,L,n+1);
t = ts(1:n);
tau1 = 0:1:L;
for j = 1:length(tau1) 
    g = exp(-a*(t-tau1(j)).^2);
    G = g.*y1';
    G = fft(G);
    gnr_spec(j,:) = abs(fftshift(G));
end
pcolor(tau1,ks,log(gnr_spec+1)')
shading interp
set(gca,'ylim',[200, 900],'Fontsize',15)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title("Sweet Child O' Mine Guitar Spectrogram")

%% Floyd
[y2, Fs] = audioread('Floyd.m4a');
tr_floyd = length(y2)/Fs;
n = length(y2);
L = tr_floyd;
k = (1/L)*[0:(n/2-1) -n/2:-1];
ks = fftshift(k);
a = 50;
ts = linspace(0,L,n+1);
t = ts(1:n);
tau2 = 0:1:L;
for j = 1:length(tau2) 
    g = exp(-a*(t-tau2(j)).^2);
    yg = g.*y2';
    ygt = fft(yg);
    floyd_spec(j,:) = abs(fftshift(ygt));
end

%% Bass Spectrogram
pcolor(tau2,ks,log(floyd_spec(:,1:end-1)+1)')
shading interp
set(gca,'ylim',[50, 200],'Fontsize',15)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title("Comfortably Numb Bass Spectrogram")

%% Guitar Spectrogram
pcolor(tau2,ks,log(floyd_spec(:,1:end-1)+1)')
shading interp
set(gca,'ylim',[200, 900],'Fontsize',15)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title("Comfortably Numb Guitar Spectrogram")