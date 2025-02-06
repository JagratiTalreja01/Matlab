clc;
clear all;

total_clusters = input('Enter the total number of clusters: '); % Total number of clusters

net_length = input('Enter the length of the network: ');
net_width = input('Enter the width of the network: ');

% Initialize node location arrays
x_loc = [];
y_loc = [];

% Energy parameters
initial_energy = 1000; % Initial energy for each node
b = 1000;
E_Tx_circuit = 50; 
E_Tx_Amp = 100; 
E_DA = 5;
n = 2; 

% Define cluster boundaries (adjust these values if necessary to fit the visual boundaries)
cluster_length = net_length / 3;
cluster_width = net_width / 2;

% Distribute nodes and select cluster heads
cluster_heads = zeros(1, total_clusters);
for cluster = 1:total_clusters
    nodes_per_cluster = input(['Enter the number of nodes in Cluster ', num2str(cluster), ': ']);
    total_nodes_cluster = nodes_per_cluster;
    row = mod(cluster - 1, 2); % 0 for first row, 1 for second row
    col = floor((cluster - 1) / 2); % 0 for first column, 1 for second column, 2 for third column
    
    % Position nodes randomly within each cluster's area
    x_loc_cluster = col * cluster_length + cluster_length * rand(1, nodes_per_cluster);
    y_loc_cluster = row * cluster_width + cluster_width * rand(1, nodes_per_cluster);
    
    x_loc = [x_loc, x_loc_cluster];
    y_loc = [y_loc, y_loc_cluster];
    
    % Calculate remaining energy for each node in the cluster
    distances = sqrt((x_loc_cluster - mean(x_loc_cluster)).^2 + (y_loc_cluster - mean(y_loc_cluster)).^2);
    node_energy = initial_energy - (E_Tx_circuit * b + E_Tx_Amp * b .* distances.^n + E_DA);
    
    % Select the cluster head based on the highest energy
    [~, idx] = max(node_energy);
    cluster_heads(cluster) = length(x_loc) - nodes_per_cluster + idx;
end

total_nodes = length(x_loc);

% Plot the network with star topology for each cluster
figure;
hold on;

% Plot cluster boundaries
for i = 0:2
    plot([i * cluster_length, i * cluster_length], [0, net_width], 'k--', 'LineWidth', 1);
end
for i = 0:1
    plot([0, net_length], [i * cluster_width, i * cluster_width], 'k--', 'LineWidth', 1);
end

% Plot nodes and cluster heads
for i = 1:total_nodes
    plot(x_loc(i), y_loc(i), 'r*', 'LineWidth', 2);
    if any(cluster_heads == i)
        plot(x_loc(i), y_loc(i), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    end
end

% Create and plot star topology within each cluster
for cluster = 1:total_clusters
    idx_start = sum(arrayfun(@(x) input(['Enter the number of nodes in Cluster ',num2str(x), ': ']), 1:cluster-1))+1;
    idx_end = sum(arrayfun(@(x) input(['Enter the number of nodes in Cluster ',num2str(x), ': ']), 1:cluster));
    cluster_head = cluster_heads(cluster);
    for i = idx_start:idx_end
        if i ~= cluster_head
            plot([x_loc(i), x_loc(cluster_head)], [y_loc(i), y_loc(cluster_head)], 'b-');
        end
    end
    
end

% Label nodes
node_labels = cellstr(num2str((1:total_nodes)'));
text(x_loc, y_loc, node_labels, 'VerticalAlignment','bottom','HorizontalAlignment','right');

% Set labels and title
xlabel('X Coordinate (meters)');
ylabel('Y Coordinate (meters)');
title('Clustered Network with Star Topology');
grid on;

% Plot and move the sink node
sink_x = x_loc(cluster_heads(1));
sink_y = y_loc(cluster_heads(1));
h_sink = plot(sink_x, sink_y, 'bs', 'MarkerSize', 10, 'LineWidth', 2);
drawnow;

for i = 1:length(cluster_heads)
    if cluster_heads(i) ~= 0
        sink_x = x_loc(cluster_heads(i));
        sink_y = y_loc(cluster_heads(i));
        set(h_sink, 'XData', sink_x, 'YData', sink_y);
        drawnow;
        pause(2);
    end
end
