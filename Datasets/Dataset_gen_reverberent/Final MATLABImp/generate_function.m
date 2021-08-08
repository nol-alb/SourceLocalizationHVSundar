function abcd = generate_function(x_src,y_src,reg,angle)
  cd 'E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master'
  mex -setup C++
  mex rir_generator.cpp rir_generator_core.cpp
  a=load('SAI_params.mat');        %here we are loading the parameters from hv sundars tdoa paper
  xcor=[];
  ycor=[];
  for i=1:8
      xcor(i)=a.Mic_pos(i,1);
      ycor(i)=a.Mic_pos(i,2);
  end

  %_____room impulse generation______%

  c = a.speed_sound;                    % Sound velocity (m/s)
  fs = 16000;                 % Sample frequency (samples/s)
  r = [a.Mic_pos(1,:) 0;a.Mic_pos(2,:) 0;a.Mic_pos(3,:) 0;a.Mic_pos(4,:) 0;a.Mic_pos(5,:) 0;a.Mic_pos(6,:) 0;a.Mic_pos(7,:) 0;a.Mic_pos(8,:) 0];    % Receiver positions [x_1 y_1 z_1 ; x_2 y_2 z_2] (m)
  L = [6 7.5 4.5];                % Room dimensions [x y z] (m) ##### sundars paper
  beta = 0.3;                 % Reverberation time (s)####### its nthere in HV sundars paper
  n = 2496;                   % Number of samples   which is 8 x 312 
  mtype = 'cardioid';    % Type of microphone
  order = -1;                 % -1 equals maximum reflection order!
  dim = 3;                    % Room dimension
  orientation = 0;            % Microphone orientation (rad)
  hp_filter = 1;              % Enable high-pass filter
  s=[];
%   s=append(s,x_src);
%   s=append(s,y_src);
%   s=append(s,2);

  s = [x_src y_src 2 ];              % Source position [x y z] (m)
  name=[x_src y_src 2 angle ];

  h0 = rir_generator(c, fs, r, s, L, beta, n, mtype, order, dim, orientation, hp_filter);

  Fs = 16000;
  [y, Fs] = audioread('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/SA1.wav');
  x01= fftfilt(h0(1,:),y);
  x02= fftfilt(h0(2,:),y);
  x03= fftfilt(h0(3,:),y);
  x04= fftfilt(h0(4,:),y);
  x05= fftfilt(h0(5,:),y);
  x06= fftfilt(h0(6,:),y);
  x07= fftfilt(h0(7,:),y);
  x08= fftfilt(h0(8,:),y);
  
  
  cd (fullfile('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/big_testdata/',reg))
  mkdir (mat2str(name))
  cd (fullfile('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/big_testdata/',reg,'/',mat2str(name)))
  

  audiowrite("audio1.wav",x01,Fs)
  audiowrite("audio2.wav",x02,Fs)
  audiowrite("audio3.wav",x03,Fs)
  audiowrite("audio4.wav",x04,Fs)
  audiowrite("audio5.wav",x05,Fs)
  audiowrite("audio6.wav",x06,Fs)
  audiowrite("audio7.wav",x07,Fs)
  audiowrite("audio8.wav",x08,Fs)
  cd (fullfile('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/big_testdata/',reg))
  abcd = 0;
end

