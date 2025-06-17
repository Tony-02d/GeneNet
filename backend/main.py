import sys
import json
import os
import time
import io
import polars as pl

from graph import build_graph, get_subgraph_by_label
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from prediction import LinkPredModel, train, test, predict_new_edges, merge_edge_labels

def debug_print(message):

    print(f"[DEBUG] {message}", file=sys.stderr, flush=True)


    try:
        with open("debug_output.log", "a") as f:
            f.write(f"[DEBUG] {message}\n")
            f.flush()
    except Exception as e:
        print(f"[ERROR] Failed to write to debug log: {e}", file=sys.stderr, flush=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def handle_request(req: dict) -> dict:
    action = req.get("action")
    parameters = req.get("parameters", {})
    dataframe_csv = req.get("dataframe_csv")

    debug_print(f"Action: {action}")

    if action == "double":
        val = parameters.get("value")
        if isinstance(val, int):
            result = val * 2
            debug_print(f"Doubled value: {val} -> {result}")
            return {"result": result}
        else:
            error_msg = f"Invalid value for double action: {val}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "get_bar_chart_data":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for bar chart data"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            key = "GeneSymbol"
            value = parameters.get("value", "")
            top_n = parameters.get("top_n", None)

            if top_n is not None:
                debug_print(f"Request for top {top_n} genes")

            debug_print(f"Request parameters - key: '{key}' (fixed to GeneSymbol), value: '{value}'")

            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")

            try:
                columns_to_read = ["GeneSymbol"]
                if value and key != "GeneSymbol":
                    columns_to_read.append(key)

                debug_print(f"Optimizing CSV reading by only loading columns: {columns_to_read}")
                try:
                    df = pl.read_csv(
                        io.StringIO(dataframe_csv),
                        columns=columns_to_read,
                        infer_schema_length=infer_schema_length,
                    )
                except Exception as e:
                    debug_print(f"Error reading specific columns: {e}, falling back to reading all columns")
                    df = pl.read_csv(
                        io.StringIO(dataframe_csv),
                        infer_schema_length=infer_schema_length,
                    )

                debug_print(f"Initial DataFrame shape: {df.shape}")

                if "GeneSymbol" in df.columns:
                    unique_count = df["GeneSymbol"].n_unique()
                    debug_print(f"Total unique genes in dataframe: {unique_count}")

                    if unique_count < 20:
                        unique_genes = df["GeneSymbol"].unique().to_list()
                        debug_print(f"Unique genes: {unique_genes}")
                    else:
                        sample_genes = df["GeneSymbol"].unique().head(10).to_list()
                        debug_print(f"First 10 genes (sample): {sample_genes}")
            except Exception as e:
                debug_print(f"Error during optimized CSV reading: {e}")
                df = pl.read_csv(
                    io.StringIO(dataframe_csv),
                    infer_schema_length=infer_schema_length,
                )
                debug_print(f"Fallback DataFrame shape: {df.shape}")

            if value:
                df_before = df.clone()
                try:
                    try:
                        numeric_value = int(value)
                        debug_print(f"Parsed value as integer: {numeric_value}")
                        debug_print(f"Numeric value provided, skipping filtering step (will use top_n instead)")
                    except ValueError:
                        debug_print(f"Value is not numeric, filtering by exact match: {value}")
                        df = df.filter(df[key] == value)
                        debug_print(f"DataFrame shape after value filtering: {df.shape}")
                        debug_print(f"Rows removed by value filtering: {df_before.height - df.height}")
                except Exception as e:
                    debug_print(f"Error during value filtering: {e}, skipping filter")
                    df = df_before

            if "GeneSymbol" in df.columns:
                debug_print("Starting grouping and aggregation...")
                start_time = time.time()

                grouped = df.group_by("GeneSymbol").agg(pl.count())
                debug_print(f"Grouped data shape: {grouped.shape}")

                end_time = time.time()
                debug_print(f"Grouping completed in {end_time - start_time:.2f} seconds")

                bar_data = {row["GeneSymbol"]: row["count"] for row in grouped.to_dicts()}

                max_genes = 30
                if top_n is not None:
                    try:
                        max_genes = int(top_n)
                        debug_print(f"Using provided top_n value: {max_genes}")
                    except (ValueError, TypeError):
                        debug_print(f"Invalid top_n value: {top_n}, using default: {max_genes}")

                if len(bar_data) > max_genes:
                    debug_print(f"Limiting result to top {max_genes} genes (out of {len(bar_data)}) to prevent lag")
                    sorted_genes = sorted(bar_data.items(), key=lambda x: x[1], reverse=True)[:max_genes]
                    bar_data = {gene: count for gene, count in sorted_genes}
                    debug_print(f"Reduced bar_data to {len(bar_data)} genes")

                debug_print("\n=== BAR CHART RESULT ===")
                debug_print(f"Total genes in result: {len(bar_data)}")

                if bar_data:
                    counts = list(bar_data.values())
                    debug_print(f"Min count: {min(counts)}")
                    debug_print(f"Max count: {max(counts)}")
                    debug_print(f"Average count: {sum(counts)/len(counts):.2f}")
                    debug_print(f"Total count sum: {sum(counts)}")

                    debug_print("\nTop 10 genes by count:")
                    sorted_genes = sorted(bar_data.items(), key=lambda x: x[1], reverse=True)
                    for i, (gene, count) in enumerate(sorted_genes[:10]):
                        debug_print(f"{i+1}. {gene}: {count}")

                debug_print(f"\nGenes being returned ({len(bar_data)}):")
                debug_print(f"{list(bar_data.keys())}")
                debug_print("=== END BAR CHART RESULT ===\n")

                sorted_genes = sorted(bar_data.items(), key=lambda x: x[1], reverse=True)
                bar_data_list = [{"gene": gene, "count": count} for gene, count in sorted_genes]

                return {"result": {"sorted_bar_data": bar_data_list}}
            else:
                error_msg = "GeneSymbol column not found in dataframe"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error processing bar chart data: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "get_subgraph_data":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for subgraph data"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            gene_symbol = parameters.get("gene_symbol", "")

            if not gene_symbol:
                error_msg = "No gene symbol provided for subgraph generation"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

            debug_print(f"Request for subgraph of gene: {gene_symbol}")

            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")

            try:
                df = pl.read_csv(
                    io.StringIO(dataframe_csv),
                    infer_schema_length=infer_schema_length,
                )

                graph = build_graph(df.to_pandas())
                subgraph = get_subgraph_by_label(graph, gene_symbol)

                result = {
                    "nodes": subgraph["nodes"],
                    "edges": subgraph["edges"],
                    "message": f"Subgraph for gene {gene_symbol} generated successfully",
                }

                return {"result": result}

            except Exception as e:
                error_msg = f"Error reading dataframe: {str(e)}"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error processing subgraph data: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "get_top_genes_graph":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for top genes graph"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            top_n = parameters.get("top_n", 10)
            debug_print(f"Request for top {top_n} genes graph")

            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")

            try:
                df = pl.read_csv(
                    io.StringIO(dataframe_csv),
                    infer_schema_length=infer_schema_length,
                )
                debug_print(f"DataFrame shape: {df.shape}")

                pandas_df = df.to_pandas()

                gene_disease_counts = pandas_df.groupby('GeneSymbol').size().reset_index(name='count')

                top_genes = gene_disease_counts.sort_values('count', ascending=False).head(top_n)

                debug_print(f"Top {len(top_genes)} genes by disease count:")
                for i, (gene, count) in enumerate(zip(top_genes['GeneSymbol'], top_genes['count'])):
                    debug_print(f"{i+1}. {gene}: {count}")

                graph = build_graph(pandas_df)

                nodes = []
                edges = []

                for gene_symbol in top_genes['GeneSymbol']:

                    gene_nodes = [n for n, data in graph.nodes(data=True) 
                                 if data.get('type') == 'gene' and data.get('label') == gene_symbol]

                    if gene_nodes:
                        gene_id = gene_nodes[0]
                        gene_data = graph.nodes[gene_id]


                        nodes.append({
                            "id": str(gene_id),
                            "label": gene_data.get('label'),
                            "type": "gene"
                        })


                        for disease_id in graph.successors(gene_id):
                            disease_data = graph.nodes[disease_id]


                            if not any(n['id'] == str(disease_id) for n in nodes):
                                nodes.append({
                                    "id": str(disease_id),
                                    "label": disease_data.get('label'),
                                    "type": "disease"
                                })


                            edges.append({
                                "source": str(gene_id),
                                "target": str(disease_id)
                            })

                result = {
                    "nodes": nodes,
                    "edges": edges,
                    "message": f"Graph for top {top_n} genes generated successfully",
                }

                debug_print(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges")

                return {"result": result}

            except Exception as e:
                error_msg = f"Error reading dataframe: {str(e)}"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error processing top genes graph: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "train_prediction_model":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for prediction model training"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            gene_symbol = parameters.get("gene_symbol", "")

            if not gene_symbol:
                error_msg = "No gene symbol provided for prediction model training"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

            debug_print(f"Training prediction model for gene: {gene_symbol}")

            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")

            try:
                df = pl.read_csv(
                    io.StringIO(dataframe_csv),
                    infer_schema_length=infer_schema_length,
                )
                debug_print(f"DataFrame shape: {df.shape}")


                pandas_df = df.to_pandas()


                graph = build_graph(pandas_df)


                subgraph_data = get_subgraph_by_label(graph, gene_symbol)


                nodes = subgraph_data["nodes"]
                edges = subgraph_data["edges"]


                node_mapping = {node["id"]: i for i, node in enumerate(nodes)}


                edge_index = []
                for edge in edges:
                    source_idx = node_mapping[edge["source"]]
                    target_idx = node_mapping[edge["target"]]
                    edge_index.append([source_idx, target_idx])

                if not edge_index:
                    error_msg = f"No edges found for gene {gene_symbol}"
                    debug_print(error_msg)
                    return {"result": None, "error": error_msg}

                edge_index = torch.tensor(edge_index, dtype=torch.long).t()


                node_types = ["gene", "disease"]
                x = []
                for node in nodes:

                    node_type_idx = node_types.index(node["type"]) if node["type"] in node_types else len(node_types)
                    one_hot = [0] * (len(node_types) + 1)
                    one_hot[node_type_idx] = 1
                    x.append(one_hot)

                x = torch.tensor(x, dtype=torch.float)

                data = Data(x=x, edge_index=edge_index)

                num_edges = edge_index.size(1)
                indices = torch.randperm(num_edges)
                train_size = int(0.8 * num_edges)

                train_indices = indices[:train_size]
                val_indices = indices[train_size:]

                train_edge_index = edge_index[:, train_indices]
                val_edge_index = edge_index[:, val_indices]

                data.pos_edge_label_index = train_edge_index
                data.pos_edge_label = torch.ones(train_edge_index.size(1))

                neg_edge_index = []
                num_nodes = x.size(0)
                existing_edges = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1)))

                while len(neg_edge_index) < train_edge_index.size(1):
                    i, j = np.random.randint(0, num_nodes, 2)
                    if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                        neg_edge_index.append([i, j])
                        existing_edges.add((i, j))

                neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long).t()
                data.neg_edge_label_index = neg_edge_index
                data.neg_edge_label = torch.zeros(neg_edge_index.size(1))

                data.val_pos_edge_label_index = val_edge_index
                data.val_pos_edge_label = torch.ones(val_edge_index.size(1))

                val_neg_edge_index = []
                while len(val_neg_edge_index) < val_edge_index.size(1):
                    i, j = np.random.randint(0, num_nodes, 2)
                    if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                        val_neg_edge_index.append([i, j])
                        existing_edges.add((i, j))

                val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).t()
                data.val_neg_edge_label_index = val_neg_edge_index
                data.val_neg_edge_label = torch.zeros(val_neg_edge_index.size(1))

                data_train = merge_edge_labels(data)

                data_val = Data(x=x, edge_index=edge_index)
                data_val.edge_label_index = torch.cat([data.val_pos_edge_label_index, data.val_neg_edge_label_index], dim=1)
                data_val.edge_label = torch.cat([data.val_pos_edge_label, data.val_neg_edge_label], dim=0)

                model = LinkPredModel(
                    input_dim=x.size(1),
                    hidden_dim=64,
                    dropout=0.3,
                    negative_slope=0.2,
                    dot_product=True,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                best_val_acc = 0
                patience = 10
                patience_counter = 0

                for epoch in range(100):
                    loss_train, acc_train, loss_val, acc_val = train(model, data_train, data_val, optimizer)

                    debug_print(f'Epoch: {epoch:03d}, Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, '
                          f'Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}')

                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        patience_counter = 0

                        model_state = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'node_mapping': node_mapping,
                            'nodes': nodes,
                            'x': x,
                            'edge_index': edge_index,
                        }
                        torch.save(model_state, f'model_{gene_symbol}.pt')
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        debug_print(f'Early stopping at epoch {epoch}')
                        break

                result = {
                    "success": True,
                    "message": f"Model for gene {gene_symbol} trained successfully with validation accuracy {best_val_acc:.4f}",
                }

                return {"result": result}

            except Exception as e:
                error_msg = f"Error processing data for prediction model: {str(e)}"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error training prediction model: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "predict_gene_links":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for gene link prediction"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            gene_symbol = parameters.get("gene_symbol", "")
            threshold = parameters.get("threshold", 0.9)

            if not gene_symbol:
                error_msg = "No gene symbol provided for gene link prediction"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

            debug_print(f"Predicting links for gene: {gene_symbol} with threshold {threshold}")


            model_path = f'model_{gene_symbol}.pt'
            if not os.path.exists(model_path):
                error_msg = f"No trained model found for gene {gene_symbol}. Please train the model first."
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")

            try:
                df = pl.read_csv(
                    io.StringIO(dataframe_csv),
                    infer_schema_length=infer_schema_length,
                )
                debug_print(f"DataFrame shape: {df.shape}")

                pandas_df = df.to_pandas()

                graph = build_graph(pandas_df)
                subgraph_data = get_subgraph_by_label(graph, gene_symbol)

                checkpoint = torch.load(model_path)

                saved_nodes = checkpoint['nodes']
                saved_node_mapping = checkpoint['node_mapping']
                saved_x = checkpoint['x']
                saved_edge_index = checkpoint['edge_index']

                model = LinkPredModel(
                    input_dim=saved_x.size(1),
                    hidden_dim=64,
                    dropout=0.3,
                    negative_slope=0.2,
                    dot_product=True,
                )

                model.load_state_dict(checkpoint['model_state_dict'])


                data = Data(x=saved_x, edge_index=saved_edge_index)


                disease_nodes = [node for node in saved_nodes if node["type"] == "disease"]
                gene_nodes = [node for node in saved_nodes if node["type"] == "gene"]

                target_gene_node = None
                for node in gene_nodes:
                    if node["label"] == gene_symbol:
                        target_gene_node = node
                        break

                if not target_gene_node:
                    error_msg = f"Gene {gene_symbol} not found in the model"
                    debug_print(error_msg)
                    return {"result": None, "error": error_msg}


                edge_label_index = []
                gene_idx = saved_node_mapping[target_gene_node["id"]]

                for disease_node in disease_nodes:
                    disease_idx = saved_node_mapping[disease_node["id"]]
                    edge_label_index.append([gene_idx, disease_idx])

                edge_label_index = torch.tensor(edge_label_index, dtype=torch.long).t()
                data.edge_label_index = edge_label_index


                new_links = predict_new_edges(model, data, threshold=threshold)


                for link in new_links:
                    source_idx = link["source"]
                    target_idx = link["target"]


                    source_id = None
                    target_id = None
                    for node_id, idx in saved_node_mapping.items():
                        if idx == source_idx:
                            source_id = node_id
                        if idx == target_idx:
                            target_id = node_id
                        if source_id and target_id:
                            break

                    link["source"] = source_id
                    link["target"] = target_id


                nodes = []
                edges = []


                nodes.append(target_gene_node)


                for link in new_links:
                    source_id = link["source"]
                    target_id = link["target"]
                    score = link["score"]


                    disease_node = None
                    for node in saved_nodes:
                        if node["id"] == target_id:
                            disease_node = node
                            break

                    if disease_node and disease_node not in nodes:
                        nodes.append(disease_node)

                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "score": score
                    })

                result = {
                    "nodes": nodes,
                    "edges": edges,
                    "message": f"Predicted {len(new_links)} links for gene {gene_symbol}",
                }

                return {"result": result}

            except Exception as e:
                error_msg = f"Error processing data for gene link prediction: {str(e)}"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error predicting gene links: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    elif action == "process_dataframe":
        if dataframe_csv is None:
            error_msg = "No dataframe provided for process_dataframe"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

        try:
            infer_schema_length = int(os.environ.get("INFER_SCHEMA_LENGTH", 1_000_000))
            debug_print(f"Reading CSV with infer_schema_length={infer_schema_length}")
            df = pl.read_csv(
                io.StringIO(dataframe_csv),
                infer_schema_length=infer_schema_length,
            )
            debug_print(f"DataFrame shape: {df.shape}")

            operation = parameters.get("operation")
            debug_print(f"Operation: {operation}")

            if operation == "count_rows":
                row_count = df.height
                debug_print(f"Row count: {row_count}")
                return {"result": row_count}

            elif operation == "get_summary":
                debug_print("Generating summary")
                summary_df = df.describe()
                summary_csv = summary_df.to_csv()
                debug_print(f"Summary shape: {summary_df.shape}")
                return {"result": "Summary generated", "dataframe_csv": summary_csv}

            else:
                error_msg = f"Unknown dataframe operation: {operation}"
                debug_print(error_msg)
                return {"result": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error processing dataframe: {str(e)}"
            debug_print(error_msg)
            return {"result": None, "error": error_msg}

    else:
        error_msg = f"Unknown action: {action}"
        debug_print(error_msg)
        return {"result": None, "error": error_msg}


def main():
    debug_print("Python backend started")
    line = sys.stdin.readline()
    if not line:
        debug_print("No input received, exiting")
        return

    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        debug_print(f"Failed to parse JSON: {e}")
        sys.exit(1)

    resp = handle_request(req)
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
    debug_print("Response sent")


if __name__ == "__main__":
    main()
