function array_des = RandomArray(min_lim, max_lim, arrey_size,rep)

% min_lim = 50;
% max_lim = 100;
% 
% arrey_size = 10;
array_des = zeros(1, arrey_size);
i = 1;
for ii= 1:arrey_size
    
    num = round((max_lim-min_lim).*rand(1,1) + min_lim,1);
    freq = round((rep-1).*rand(1,1) + 1);
    
    for j = 1:freq
        
       array_des(i) = num;
       i = i + 1;
    end
end