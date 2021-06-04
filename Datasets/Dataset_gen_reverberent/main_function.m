cd 'C:/Users/Niraj/Desktop/HVsundarSourceLocalization/Datasets/Dataset_gen_reverberent'
b=matfile('C:/Users/Niraj/Desktop/HVsundarSourceLocalization/Datasets/Dataset_gen_reverberent/outCoordinates.mat');
regions=["R0" "R1" "R2" "R3" "R4" "R5" "R6" "R47"]
f=fieldnames(b)

for k=2:9
    mkdir (fullfile('C:/Users/Niraj/Desktop/HVsundarSourceLocalization/Datasets/Dataset_gen_reverberent/testdata/',regions(k-1)))
    a=b.(f{k})
    for i =1:50
        xx=a(i,1);
        yy=a(i,2);
        xyz=generate_function(xx,yy,regions(k-1));
    end
    cd ('C:/Users/Niraj/Desktop/HVsundarSourceLocalization/Datasets/Dataset_gen_reverberent/testdata')

end
