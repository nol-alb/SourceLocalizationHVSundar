cd 'E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master'
b=matfile('E:/Final year project/datasets/HVsundar_dataset/outCoordinates.mat');
regions=["R8" "R4" "R2" "R7" "R6" "R1" "R5" "R3"]
f=fieldnames(b)
center = [3 3.75 2];
dmax=2;
dmin=0.0110;
% f=orderfields(f)
for k=2:9
    mkdir (fullfile('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/big_testdata/',regions(k-1)))
    a=b.(f{k})
%     csvwrite('coarse.csv',0);
    for i =1:200
        xx=a(i,1);
        yy=a(i,2);
        custom(1)=xx; %X coordinate of custom
        custom(2)=yy; %Y coordinate of custom
        custom(3)=2;
%         new_custom_x=[new_custom_x custom(1)];
%         new_custom_y=[new_custom_y custom(2)];
        r=sqrt((center(1)-custom(1))^2+(center(2)-custom(2))^2);
        straight(1)=center(1);
        straight(2)=center(2)+r;
        straight(3)=2;
        chord=sqrt((straight(1)-custom(1))^2+(straight(2)-custom(2))^2);
        alpha=acosd((2*r^2-chord^2)/(2*r^2));
        if (custom(1)>center(1))
            alpha=360-alpha;
        end
          xyz=generate_function(xx,yy,regions(k-1),alpha);
        
        
            if(alpha<45)
                enc_a=(alpha)/(45);
            elseif(alpha>45 && alpha<90)
                enc_a=(alpha-45)/(45);
            elseif(alpha>90 && alpha<135)
                enc_a=(alpha-90)/(45);
            elseif(alpha>135 && alpha<180)
                enc_a=(alpha-135)/(45);
            elseif(alpha>180 && alpha<225)
                enc_a=(alpha-180)/(45);
            elseif(alpha>225 && alpha<270)
                enc_a=(alpha-225)/(45);
            elseif(alpha>270 && alpha<315)
                enc_a=(alpha-270)/(45);
            else(alpha>315)
                enc_a=(alpha-315)/(45);
            end
            enc_d=(r-dmin)/(dmax-dmin);
        
        
    end
      cd ('E:/Final year project/datasets/HVsundar_dataset/RIR-Generator-master/big_testdata')

end