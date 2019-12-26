function graph_data = process_data(folder_name, dataset_name, do_save)
%PROCESS_DATA Load graph data into MALTAB.
%   folder_name - Name of the folder that stores the data.
%   dataset_name - Name of the dataset.
%   do_save - Set to true to save the processed data as a .mat file.
%Returns:
%   graph_data - A struct containing the following fiels:
%       n_graphs - Number of graphs.
%       graphs - A struct array whose size is the number of graphs. It
%                contains the following fields:
%                   n_nodes - Number of nodes.
%                   n_edges - Number of edges.
%                   A - Adjacent matrix (stored as a sparse matrix, use
%                       full() if you want the dense version).
%                   label - Label of the graph.
%                   node_labels - A vector of node labels.
%       n_classes - Number of classes.
%       class_labels - A arrays of all possible labels.
%       n_attributes_per_node - Number of attributes per node.
%       unique_node_labels - A vector of all possible node labels.
%       has_node_labels - Indicates whether node labels exist.
%       has_node_attributes - Indicates whether node attributes exist.
if nargin < 2
    dataset_name = folder_name;
end
if nargin < 3
    do_save = false;
end
edge_definition_file_name = fullfile(folder_name, [dataset_name '_A.txt']);
graph_indicator_file_name = fullfile(folder_name, [dataset_name '_graph_indicator.txt']);
graph_label_file_name = fullfile(folder_name, [dataset_name '_graph_labels.txt']);
node_label_file_name = fullfile(folder_name, [dataset_name '_node_labels.txt']);
node_attributes_file_name = fullfile(folder_name, [dataset_name '_node_attributes.txt']);

% load graph labels
graph_labels = csvread(graph_label_file_name);
n_graphs = length(graph_labels);
labels = unique(graph_labels);
n_classes = length(labels);

% prepare struct
graphs = repmat(struct('n_nodes', 0, 'n_edges', 0, 'A', [], 'ConEdge', [], 'label', [], ...
                       'node_labels', [], 'node_attributes', []), n_graphs, 1);

% load edges
edge_data = csvread(edge_definition_file_name);

% load graph indicators
graph_indicators = csvread(graph_indicator_file_name);

% load node labels
if exist(node_label_file_name, 'file')
    node_label_data = csvread(node_label_file_name);
else
    node_label_data = [];
end

% load node attributes
if exist(node_attributes_file_name, 'file')
    node_attributes_data = csvread(node_attributes_file_name);
else
    node_attributes_data = [];
end

% process graphs
n_total_nodes = length(graph_indicators);
A_global = sparse(edge_data(:,1), edge_data(:,2), 1, n_total_nodes, n_total_nodes);
A_global(A_global > 0) = 1;
for ii = 1:n_graphs
    cur_nodes = find(graph_indicators == ii);
    n_nodes = length(cur_nodes);
    graphs(ii).n_nodes = n_nodes;
    graphs(ii).label = graph_labels(ii);
    % build adjacent matrix
    A = A_global(cur_nodes, cur_nodes);
    if any(A' ~= A)
        error('Adjacent matrix is not symmetric.');
    end
    graphs(ii).A = A;
    graphs(ii).n_edges = nnz(A) / 2;
    AdjMat=A-diag(diag(A))+eye(n_nodes); Deg=AdjMat*ones(n_nodes,1); 
    MaxDeg=max(Deg);
    EdgConnectMat=zeros(n_nodes,MaxDeg);
    for i=1:n_nodes
        EdgInd=find(AdjMat(i,:));
        %EdgInd=EdgInd-ones(size(EdgInd));
        EdgConnectMat(i,1:length(EdgInd))=EdgInd;
    end
    graphs(ii).ConEdge=sparse(EdgConnectMat);
    % assign node labels
    if ~isempty(node_label_data)
        graphs(ii).node_labels = node_label_data(cur_nodes);
    end
    if ~isempty(node_attributes_data)
        graphs(ii).node_attributes = node_attributes_data(cur_nodes,:);
    end
end

graph_data.n_graphs = n_graphs;
graph_data.n_classes = n_classes;
graph_data.n_attributes_per_node = size(node_attributes_data, 2);
graph_data.unique_node_labels = unique(node_label_data);
graph_data.class_labels = labels;
graph_data.graphs = graphs;
graph_data.has_node_labels = ~isempty(node_label_data);
graph_data.has_node_attributes = ~isempty(node_attributes_data);
if do_save
    save_data = struct();
    save_data.([dataset_name '_data']) = graph_data;
    save(fullfile(dataset_name), '-struct', 'save_data');
end
end

