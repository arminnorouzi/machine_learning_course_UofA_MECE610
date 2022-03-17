function [data_n, min_data, max_data] = dataTrainStandardized(data)
min_data = min(data);
max_data = max(data);

data_n = (data - min_data) / (max_data-min_data);
end

