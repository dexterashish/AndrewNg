C = 0.01;
sigma = 0.03;
error_min = 1000000;
count = 0;
sigma_check = sigma;
for i=1:8
  for j=1:8
    C
    sigma
    sigma = sigma*10;
    count = count + 1;
  endfor
  sigma = sigma_check;
  C = C*10;
endfor

error_min
count