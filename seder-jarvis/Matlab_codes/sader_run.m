% z = [0.00e+00 ,5.00e-12 ,1.00e-11 ,1.50e-11 ,2.00e-11 ,2.50e-11 ,3.00e-11 ,3.50e-11 ,4.00e-11 ,4.50e-11 ,5.00e-11 ,5.50e-11 ,6.00e-11 ,6.50e-11 ,7.00e-11 ,7.50e-11 ,8.00e-11 ,8.50e-11 ,9.00e-11 ,9.50e-11 ,1.00e-10 ,1.05e-10 ,1.10e-10 ,1.15e-10 ,1.20e-10 ,1.25e-10 ,1.30e-10 ,1.35e-10 ,1.40e-10 ,1.45e-10 ,1.50e-10 ,1.55e-10 ,1.60e-10, 1.65e-10, 1.70e-10 ,1.75e-10, 1.80e-10, 1.85e-10, 1.90e-10, 1.95e-10 ,2.00e-10, 2.05e-10, 2.10e-10 ,2.15e-10, 2.20e-10, 2.25e-10, 2.30e-10, 2.35e-10,2.40e-10, 2.45e-10, 2.50e-10 ,2.55e-10, 2.60e-10, 2.65e-10, 2.70e-10, 2.75e-10,2.80e-10, 2.85e-10, 2.90e-10 ,2.95e-10, 3.00e-10, 3.05e-10, 3.10e-10, 3.15e-10,3.20e-10, 3.25e-10, 3.30e-10, 3.35e-10, 3.40e-10, 3.45e-10, 3.50e-10, 3.55e-10,3.60e-10, 3.65e-10, 3.70e-10, 3.75e-10, 3.80e-10, 3.85e-10, 3.90e-10, 3.95e-10 ,4.00e-10, 4.05e-10, 4.10e-10, 4.15e-10, 4.20e-10, 4.25e-10, 4.30e-10, 4.35e-10,4.40e-10, 4.45e-10, 4.50e-10, 4.55e-10, 4.60e-10, 4.65e-10, 4.70e-10, 4.75e-10 ,4.80e-10, 4.85e-10, 4.90e-10, 4.95e-10, 5.00e-10];
% delta_f = [-9.92785644e-01 ,-1.04128738e+00, -1.09566400e+00, -1.14069661e+00,-1.19146647e+00, -1.24126820e+00, -1.29074357e+00, -1.33392674e+00,-1.37982949e+00, -1.41562090e+00, -1.44996319e+00, -1.46727993e+00, -1.48214032e+00, -1.47482768e+00, -1.44778437e+00, -1.40508415e+00, -1.35181563e+00, -1.27378928e+00, -1.18897405e+00, -1.09526628e+00, -9.93097534e-01, -8.94459822e-01, -8.00798960e-01, -7.09304383e-01, -6.20987975e-01, -5.47895067e-01, -4.78702580e-01, -4.23073379e-01, -3.74740889e-01, -3.20738961e-01, -2.87697626e-01, -2.49945276e-01, -2.20808616e-01, -1.98505339e-01, -1.69019090e-01, -1.49952659e-01, -1.37595678e-01, -1.17545444e-01, -1.11551967e-01, -9.23504266e-02,-8.17034473e-02, -8.14423136e-02, -6.86950041e-02, -6.24929812e-02, -5.86070646e-02, -5.09952417e-02, -4.87335124e-02, -3.61442471e-02, -3.95548257e-02, -3.16607242e-02, -3.30398344e-02 ,-2.81115435e-02,-2.89700250e-02, -2.06400572e-02, -1.63820541e-02 ,-1.56052005e-02, -1.64341970e-02, -1.50897733e-02, -1.83255809e-02 ,-1.74884221e-02, -1.40138688e-02, -8.18063067e-03, -4.69634784e-03 ,-1.19192707e-02, -9.30585258e-03, -5.11868047e-03, -6.77960728e-03, -3.48061273e-03, -8.79205458e-03, -6.91982582e-03, -6.12512559e-03, -1.20483190e-03,  6.60685526e-04, -1.98498739e-03,  3.22151959e-03, -4.05685272e-03,  1.12879614e-03,  2.37736173e-03, -4.66075487e-03, -8.90174349e-04, -3.21693720e-03, -1.80355765e-03, -4.23028163e-03, -4.10966567e-05,  5.30573607e-03,  1.93720723e-03, -3.00972458e-03, -7.12306815e-04, 2.83520208e-03,  6.02910463e-03,  3.43421337e-03,  3.07062448e-03,  2.51923967e-03,  4.38656095e-03,  1.50246794e-03,  6.68741404e-03,  4.70580501e-03, -3.76615399e-04,  1.90921683e-03,  4.72312547e-03,7.88270952e-03];

filename = 'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\outputed_mathematica_data\Z-Spectroscopy_BP_00188_z.txt';

fileID = fopen(filename, 'r');

if fileID == -1
    error('File could not be opened');
end

data = textscan(fileID, '%f %f', 'Delimiter', ',', 'HeaderLines', 1);

fclose(fileID);

z = data{1};
delta_f = data{2};

A = 1e-10;
k = 2000;
f_0 = 20000;

%set delta_f to the same length as z
[z_feven, f_feven] = Feven_deconv(z, delta_f, A, 5, 2, 0);
display([numel(z), numel(delta_f)])
[f, z , delta_f] = saderF(z, delta_f, f_0, k, A);






% display(f)

% Assuming z and f are already defined
% display(f)

% Specify the output filename
output_filename = 'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\matlab_z_f_output.txt';

% Open the file for writing
output_fileID = fopen(output_filename, 'w');

% Check if the file opened successfully
if output_fileID == -1
    error('Output file could not be opened');
else
    disp(['File opened successfully: ', output_filename]);
end

% Write header
fprintf(output_fileID, 'z f * 1e9 \n');

% Write data
for i = 1:length(z)
    fprintf(output_fileID, '%f %f\n', real(z(i)), real(f(i)));
end

% Close the file
fclose(output_fileID);

% Confirm file writing
disp(['Data written to file: ', output_filename]);

% Plot the data
plot(z, f, 'o');
hold on;
plot(z_feven, f_feven, 'o');
title('z vs f');
xlabel('z');
ylabel('f');
legend('Original data', 'Feven deconvoluted data');
hold off;