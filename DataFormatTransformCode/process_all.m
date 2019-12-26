% remember to replace - to _ in file names!
datasets={'PTC_FM','MUTAG'};
for ii = 1:length(datasets)
    process_data(datasets{ii}, datasets{ii}, true);
end