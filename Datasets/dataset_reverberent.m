cd /Users/noelalben/RIR-Generator
mex -setup C++
mex rir_generator.cpp rir_generator_core.cpp
a=load('SAI_params.mat');        %here we are loading the parameters from hv sundars tdoa paper
xcor=[];
ycor=[];
for i=1:8
    xcor(i)=a.Mic_pos(i,1);
    ycor(i)=a.Mic_pos(i,2);
end
figure(1)
plot(xcor,ycor);                 %visualizing the microphone array
%______________room impulse generation_______________%

c = a.speed_sound;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
r = [a.Mic_pos(1,:) 0;a.Mic_pos(2,:) 0;a.Mic_pos(3,:) 0;a.Mic_pos(4,:) 0;a.Mic_pos(5,:) 0;a.Mic_pos(6,:) 0;a.Mic_pos(7,:) 0;a.Mic_pos(8,:) 0];    % Receiver positions [x_1 y_1 z_1 ; x_2 y_2 z_2] (m)
L = [6 7.5 4.5];                % Room dimensions [x y z] (m) ##### sundars paper
beta = 0.3;                 % Reverberation time (s)####### its nthere in HV sundars paper
n = 2496;                   % Number of samples   which is 8 x 312 
mtype = 'hypercardioid';    % Type of microphone
order = -1;                 % -1 equals maximum reflection order!
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter


s = [4 3 2];              % Source position [x y z] (m)

h0 = rir_generator(c, fs, r, s, L, beta, n, mtype, order, dim, orientation, hp_filter);

Fs = 16000;
[y, Fs] = audioread('/Users/noelalben/Desktop/VTU_PROJ/SA1.wav');
x01= fftfilt(h0(1,:),y);
x02= fftfilt(h0(2,:),y);
x03= fftfilt(h0(3,:),y);
x04= fftfilt(h0(4,:),y);
x05= fftfilt(h0(5,:),y);
x06= fftfilt(h0(6,:),y);
x07= fftfilt(h0(7,:),y);
x08= fftfilt(h0(8,:),y);

X0 = [x01, x02, x03, x04, x05, x06, x07, x08];

c = a.speed_sound;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
r = [a.Mic_pos(1,:) 0;a.Mic_pos(2,:) 0;a.Mic_pos(3,:) 0;a.Mic_pos(4,:) 0;a.Mic_pos(5,:) 0;a.Mic_pos(6,:) 0;a.Mic_pos(7,:) 0;a.Mic_pos(8,:) 0];    % Receiver positions [x_1 y_1 z_1 ; x_2 y_2 z_2] (m)
L = [6 7.5 4.5];                % Room dimensions [x y z] (m) ##### sundars paper
beta = 0.3;                 % Reverberation time (s)####### its nthere in HV sundars paper
n = 2496;                   % Number of samples   which is 8 x 312 
mtype = 'hypercardioid';    % Type of microphone
order = -1;                 % -1 equals maximum reflection order!
dim = 3;                    % Room dimension
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter


s = [4 2.5 2];              % Source position [x y z] (m)

h = rir_generator(c, fs, r, s, L, beta, n, mtype, order, dim, orientation, hp_filter);
figure(2)


[y, Fs] = audioread('/Users/noelalben/Desktop/VTU_PROJ/SA1.wav');
x11= fftfilt(h(1,:),y);
x12= fftfilt(h(2,:),y);
x13= fftfilt(h(3,:),y);
x14= fftfilt(h(4,:),y);
x15= fftfilt(h(5,:),y);
x16= fftfilt(h(6,:),y);
x17= fftfilt(h(7,:),y);
x18= fftfilt(h(8,:),y);

Xm1 = x11+x01;
Xm2 = x12+x02;
Xm3 = x13+x03;
Xm4 = x14+x04;
Xm5 = x15+x05;
Xm6 = x16+x06;
Xm7 = x17+x07;
Xm8 = x18+x08;

X = [4.8478    1.1522]
X1 = [3.7654    2.2346]
X2 = [2.2346    3.7654]
X3 = [1.1522    4.8478]

Y = [4.5154    2.9846]
Y1 = [5.5978    1.9022]
Y2 = [5.5978    1.9022]
Y3 = [4.5154    2.9846]



plot(X,Y)
hold on
plot(X,Y,'r*')
plot(X1,Y1)
hold on
plot(X1,Y1,'r*')
plot(X2,Y2)
hold on 
plot(X2,Y2,'r*')
plot(X3,Y3)
plot(X3,Y3,'r*')
plot(xcor,ycor)
hold on
plot(x1cor,y1cor)
plot(3.8,3.75,'ko')
hold on
plot(3.8,2.75,'ko')



player1 = audioplayer(x02,Fs);
player2 = audioplayer(x12,Fs);
player3 = audioplayer(Xm2,Fs);
player5 = audioplayer(x05,Fs);








