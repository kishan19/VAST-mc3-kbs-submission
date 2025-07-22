import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from dash.dependencies import Input, Output, State
import os
from pyvis.network import Network
import tempfile
from collections import Counter
import networkx as nx
from collections import defaultdict
from dash import dash_table
import uuid

# LangGraph and LangChain imports for chatbot
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver


# =====================
# Data Loader Section
# =====================
# Place all data loading and preprocessing here for easy editing/customization
comms_df = pd.read_csv("data/MC3_data_no_pseudonyms.csv")
# print(comms_df.head())
sunburst_packet_df = pd.read_csv("data/sunburst_packet_df.csv")
# print(sunburst_packet_df.head())
topic_model_df=pd.read_csv("data/topic_model.csv")

embeddings_df=pd.read_csv("data/message_embeddings.csv")

nadia_df=pd.read_csv("data/nadia_conti.csv")

pseduonyms_df=pd.read_csv("data/pseduonyms_finder.csv")
pseduonyms_df['potential_pseduonym_id']=pseduonyms_df.index+1

# =====================
# Chatbot Logic Setup
# =====================
def setup_chatbot_logic():
    """
    Setup the chatbot logic using LangGraph and SQL database
    Returns: react_graph_memory, config
    """
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    import uuid
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import SQLDatabase
    from sqlalchemy import create_engine
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langgraph.graph import MessagesState
    from langgraph.graph import START, StateGraph, END
    from langgraph.prebuilt import tools_condition, ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    
    # Load environment variables
    #load_dotenv(dotenv_path="secret/.env")
    try:
        load_dotenv(dotenv_path="secret/.env")
        api_key = os.getenv("OPENAI_API_KEY")
    except Exception as ex:
        api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    
    # Load and prepare data
    chatbot_df = pd.read_csv("data/topic_model.csv")
    chatbot_df.drop(columns=['content', 'mins_since_start_of_day'], inplace=True)
    
    # Create SQL database
    engine = create_engine("sqlite:///mc3.db")
    chatbot_df.to_sql("mc3data", engine, index=False, if_exists='replace')



    chatbot_df.to_sql("mc3data", engine, index=False, if_exists='replace')
    
    # Setup database toolkit
    db = SQLDatabase(engine=engine)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # System prompt
    system_prompt = f"""
    You are a smart agent designed to interface with a SQLite database of boat radio communications.
    Each communication entry includes a timestamp, timeblock, date, source, target, identified entities, topics, clusters, and other attributes.

    Your job is to:
    - Convert a user's natural language question into a syntactically correct SQLite query.
    - Execute the query using the tools available.
    - Interpret the results and answer the user's question clearly.
    - Highlight relevant temporal patterns (hours, dates) and related entities when possible.

    Guidelines:
    1. Only query **relevant columns** based on the question.
    2. NEVER use `SELECT *` — choose only what's necessary.
    3. Prioritize temporal ordering (`ORDER BY timestamp` or `time`) if the question implies time.
    4. For vague names, use approximate match tools (`LIKE` or entity similarity if available).
    5. NEVER write `INSERT`, `UPDATE`, or `DELETE` queries.
    6. If a query fails, correct and retry after double-checking the syntax.

    You have access to the following table(s): {db.get_usable_table_names()}.

    After executing the query, summarize the results in context. Discuss any temporal patterns and relevant entities involved.
    """
    
    system_message = SystemMessage(content=system_prompt)
    
    # Define extended state
    class ExtendedMessagesState(MessagesState):
        use_sql: bool
    
    # Process conversation function
    def process_conversation(state: ExtendedMessagesState):
        final_output = {"messages": llm_with_tools.invoke([system_message] + state["messages"])}
        return final_output
    
    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("conversation", process_conversation)
    builder.add_node("tools", ToolNode(tools))
    builder.add_conditional_edges("conversation", tools_condition)
    builder.add_edge("tools", "conversation")
    builder.add_edge(START, "conversation")
    builder.add_edge("conversation", END)
    
    # Compile with memory
    memory = MemorySaver()
    react_graph_memory = builder.compile(checkpointer=memory)
    
    # Create config
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    return react_graph_memory, config

def get_chatbot_response(question, react_graph_memory, config):
    """
    Get response from chatbot for a given question
    Args:
        question: User's question
        react_graph_memory: Compiled LangGraph with memory
        config: Configuration with thread_id
    Returns:
        str: Chatbot response
    """
    try:
        responses = react_graph_memory.invoke({"messages": [question]}, config)
        return responses['messages'][-1].content
    except Exception as e:
        return f"Error processing your question: {str(e)}"

# Initialize chatbot components
chatbot_graph, chatbot_config = setup_chatbot_logic()

# Session management for chatbot
chatbot_session_count = 0
MAX_MESSAGES_PER_SESSION = 5

# =====================
# Sunburst Chart Helper
# =====================
# Initialize selected_entities_sunburst to default (empty list)
def build_sunburst_figure(selected_entities_sunburst):
    all_time_slots = pd.date_range("08:00", "15:00", freq="15min").strftime("%H:%M").tolist()[:-1][::-1]
    labels = []
    parents = []
    ids = []
    colors = []
    customdata = []
    hovertemplate = []
    gray_color = "lightgray"
    dates = sorted(sunburst_packet_df['date'].unique(), reverse=True)
    palette = plotly.colors.qualitative.Plotly
    while len(palette) < len(dates):
        palette += palette
    date_color_map = {date: palette[i] for i, date in enumerate(dates)}
    labels.append("root")
    parents.append("")
    ids.append("root")
    colors.append("white")
    customdata.append("")
    hovertemplate.append("<b>All Dates</b>")
    for date in dates:
        labels.append(str(date))
        parents.append("root")
        ids.append(str(date))
        colors.append(date_color_map[date])
        customdata.append("")
        hovertemplate.append(f"<b>Date:</b> {date}")
        df_date = sunburst_packet_df[sunburst_packet_df['date'] == date]
        pair_time_map = {}
        for _, row in df_date.iterrows():
            slots = [s.strip() for s in row['Time block'].strip('[]').split(',')]
            pair = row['Pair']
            if pair not in pair_time_map:
                pair_time_map[pair] = []
            pair_time_map[pair].extend(slots)
        slot_to_pairs = {slot: [] for slot in all_time_slots}
        for pair, slots in pair_time_map.items():
            for slot in slots:
                if slot in slot_to_pairs:
                    slot_to_pairs[slot].append(pair)
        for slot in all_time_slots:
            pairs_in_slot = slot_to_pairs[slot]
            if pairs_in_slot:
                for pair in sorted(set(pairs_in_slot), reverse=True):
                    # Check if this pair involves any selected entity
                    if selected_entities_sunburst is not None and len(selected_entities_sunburst) > 0:
                        pair_entities = [p.strip() for p in pair.replace('<->', '->').replace('->', '->').split('->')]
                        if any(e in pair_entities for e in selected_entities_sunburst):
                            node_color = date_color_map[date]
                        else:
                            node_color = "black"
                    else:
                        node_color = date_color_map[date]
                    labels.append(pair)
                    parents.append(str(date))
                    ids.append(f"{date}-{slot}-{pair}")
                    colors.append(node_color)
                    customdata.append(f"{pair}|{slot}")
                    hovertemplate.append(f"<b>Pair:</b> {pair}<br><b>Time Slot:</b> {slot}")
            else:
                labels.append(f"no comms")
                parents.append(str(date))
                ids.append(f"{date}-{slot}-no-comms")
                colors.append(gray_color)
                customdata.append(slot)
                hovertemplate.append(f"<b>No comms</b><br><b>Time Slot:</b> {slot}")
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        ids=ids,
        branchvalues="total",
        insidetextorientation='radial',
        marker=dict(colors=colors),
        customdata=customdata,
        hovertemplate=hovertemplate,
        sort=False
    ))
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        title="Entities (Pairs) by Date with Clockwise Time Slots (No Comms in Gray; Filtered in Black)",
        width=None,
        height=800,
    )
    return fig

# =====================
# PyVis Network Helper
# =====================
def build_entity_network(df, topic_model_df, min_group_size=3):
    net = Network(height="700px", width="100%", notebook=False, directed=True)
    
    # Define color palette for entity types
    entity_type_colors = {
        'Person': '#0074D9',      # Blue for people
        'Vessel': '#3D9970',      # Green for vessels
        'Location': '#FFDC00',    # Yellow for locations
        'Organization': '#FF4136', # Red for organizations
        'Other': '#AAAAAA',       # Gray for others
        'uncategorized': '#000000', # Black for uncategorized
        'mentioned_entity': '#000000' # Black for mentioned entities
    }
    # Assign a unique color to each topic
    topic_names = topic_model_df['topic'].unique().tolist()
    topic_palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
    ]
    topic_color_map = {topic: topic_palette[i % len(topic_palette)] for i, topic in enumerate(topic_names)}
    
    # Parse Entities field to understand who is talking about whom to whom
    entity_mentions = Counter()
    # AGGREGATE: (source, entity, target) -> count
    communication_patterns = Counter()  # (source, entity, target) -> count
    entity_types = {}  # Store entity types for coloring
    source_communication_count = Counter()  # Count communications sent by each source
    target_communication_count = Counter()  # Count communications received by each target
    node_topics = {}  # Store topic for each node
    
    # Build a lookup for Message_ID to topic
    msgid_to_topic = dict(zip(topic_model_df['Message_ID'], topic_model_df['topic']))
    
    for _, row in df.iterrows():
        try:
            # Get source, target, and mentioned entities
            source = row['source'] if 'source' in row else row['Pair'].split('<')[0].strip()
            target = row['target'] if 'target' in row else row['Pair'].split('>')[-1].strip()
            message_id = row.get('Message_ID', None)
            topic = msgid_to_topic.get(message_id, None)
            
            # Count communications sent by source and received by target
            source_communication_count[source] += 1
            target_communication_count[target] += 1
            
            # Store entity types for coloring
            source_type = row.get('source_entity_type', 'uncategorized')
            target_type = row.get('target_entity_type', 'uncategorized')
            entity_types[source] = source_type
            entity_types[target] = target_type
            
            # Parse Entities field - it's usually a string representation of a list
            entities_str = str(row['Entities'])
            if entities_str.startswith('[') and entities_str.endswith(']'):
                # Remove brackets and split by comma
                entities_str = entities_str[1:-1]
                mentioned_entities = [e.strip().strip("'\"") for e in entities_str.split(',') if e.strip()]
            else:
                # Fallback: split by comma if not in list format
                mentioned_entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            
            # Count communication patterns: source mentions entity to target (AGGREGATED)
            for entity in mentioned_entities:
                if entity and entity.lower() != 'nemo reef':  # Skip empty or irrelevant entities
                    entity_mentions[entity] += 1
                    communication_patterns[(source, entity, target)] += 1
                    # Categorize mentioned entities as "mentioned_entity"
                    if entity not in entity_types:
                        entity_types[entity] = 'mentioned_entity'
                    # Store topic for mentioned entity
                    if entity not in node_topics:
                        node_topics[entity] = topic
            # Store topic for source and target
            if source not in node_topics:
                node_topics[source] = topic
            if target not in node_topics:
                node_topics[target] = topic
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue
    
    # Add nodes for sources, targets, and mentioned entities
    added_nodes = set()
    
    # Add source and target nodes (fixed size, colored by entity type, topic border)
    FIXED_NODE_SIZE = 25
    for (source, entity, target) in communication_patterns:
        if source not in added_nodes:
            source_type = entity_types.get(source, 'uncategorized')
            color = entity_type_colors.get(source_type, entity_type_colors['uncategorized'])
            border_color = 'red'
            net.add_node(
                source,
                label=source,
                size=FIXED_NODE_SIZE,
                color={"border": border_color, "background": color},
                borderWidth=6,
                borderWidthSelected=8,
                title=f"Source: {source} ({source_type})\nTopic: {node_topics.get(source, 'Unknown')}",
                font={"color": "black", "background": "yellow", "size": 24, "face": "Comic Sans MS"},
                physics=True
            )
            added_nodes.add(source)
        if target not in added_nodes:
            target_type = entity_types.get(target, 'uncategorized')
            color = entity_type_colors.get(target_type, entity_type_colors['uncategorized'])
            border_color = 'red'
            net.add_node(
                target,
                label=target,
                size=FIXED_NODE_SIZE,
                color={"border": border_color, "background": color},
                borderWidth=6,
                borderWidthSelected=8,
                title=f"Target: {target} ({target_type})\nTopic: {node_topics.get(target, 'Unknown')}",
                font={"color": "black", "background": "yellow", "size": 24, "face": "Comic Sans MS"},
                physics=True
            )
            added_nodes.add(target)
    
    # Add entity nodes (smaller size, black color, categorized as mentioned_entity, topic border)
    MENTIONED_ENTITY_SIZE = 16
    for entity, freq in entity_mentions.items():
        if entity not in added_nodes:
            border_color = topic_color_map.get(node_topics.get(entity, None), '#cccccc')
            net.add_node(entity, label=entity, size=MENTIONED_ENTITY_SIZE, color='#000000', borderWidth=4, borderWidthSelected=6,
                        title=f"Mentioned Entity: {entity} (mentioned {freq} times)\nTopic: {node_topics.get(entity, 'Unknown')}",
                        border=border_color, 
                        physics=True)
            added_nodes.add(entity)
    
    # Add edges showing direct communications between source and target only
    direct_communication_count = Counter()  # (source, target) -> count
    for (source, entity, target), count in communication_patterns.items():
        direct_communication_count[(source, target)] += count

    for (source, target), count in direct_communication_count.items():
        width = 1 + min(count // 2, 10)  # scale for visibility
        color = '#000000'  # All edges black
        net.add_edge(source, target, value=count, width=width, color=color,
                    title=f"{source} sent {count} messages to {target}")
    
    # Add dotted gray lines from source/target to mentioned entity nodes, with full context in tooltip
    # AGGREGATED: Only one edge per (source, entity, target)
    for (source, entity, target), count in communication_patterns.items():
        if entity in added_nodes:
            # From source to mentioned entity (show to whom and count)
            net.add_edge(
                source,
                entity,
                width=2,
                color='#888888',
                dashes=True,
                title=f"{source} mentions {entity} to {target} {count} times."
            )
            # From target to mentioned entity (show from whom and count)
            net.add_edge(
                target,
                entity,
                width=2,
                color='#888888',
                dashes=True,
                title=f"{target} mentions {entity} (as target) from {source} {count} times."
            )
    
    # Physics for better layout and spacing (less shaky)
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=300, spring_strength=0.005, damping=0.25, overlap=0.8)
    return net

def get_network_html(net, topic_color_map):
    """Generate HTML string for the network with color legend and topic legend"""
    try:
        html_content = net.generate_html()
        # Add color legend to the HTML
        legend_html = """
        <div style=\"position: absolute; top: 24px; right: 24px; z-index: 9999; background: white; padding: 18px 32px; border: 3px solid #bbb; border-radius: 32px; font-family: Arial, sans-serif; font-size: 22px; display: flex; flex-direction: row; flex-wrap: wrap; align-items: center; justify-content: center; gap: 32px; box-shadow: 0 4px 24px rgba(0,0,0,0.08);\">
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #0074D9; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Person</span></div>
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #3D9970; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Vessel</span></div>
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #FFDC00; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Location</span></div>
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #FF4136; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Organization</span></div>
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #000000; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Mentioned Entity</span></div>
            <div style=\"display: flex; align-items: center; gap: 12px;\"><div style=\"width: 32px; height: 32px; background: #AAAAAA; border-radius: 8px;\"></div><span style=\"font-weight: 500;\">Other</span></div>
        </div>
        """
        # Insert legend into the HTML
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', legend_html + '</body>')
        else:
            html_content += legend_html
        return html_content
    except Exception as e:
        # Fallback HTML if PyVis fails
        return f"""
        <html>
        <head>
            <title>Network Visualization</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h2>Network Visualization</h2>
            <p class=\"error\">Error generating network: {str(e)}</p>
            <p>Please check your data and PyVis installation.</p>
        </body>
        </html>
        """

# =====================
# Community Subplots by Topic
# =====================
def create_topic_community_subplots():
    import networkx as nx
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    df = pd.read_csv('data/topic_model.csv')
    topics = df['topic'].unique().tolist()
    topics = [t for t in topics if pd.notnull(t)]
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=topics)
    topic_colors = plotly.colors.qualitative.Plotly
    for idx, topic in enumerate(topics):
        topic_df = df[df['topic'] == topic]
        G = nx.Graph()
        for _, row in topic_df.iterrows():
            src = row['source_entity']
            tgt = row['target_entity']
            if pd.notnull(src) and pd.notnull(tgt):
                G.add_edge(src, tgt)
        pos = nx.spring_layout(G, seed=42, k=1.5) if len(G) > 1 else {n: (0, 0) for n in G.nodes()}
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                color=topic_colors[idx % len(topic_colors)],
                size=18,
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            showlegend=False
        )
        edge_x, edge_y = [], []
        for src, tgt in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.add_trace(edge_trace, row=row, col=col)
        fig.add_trace(node_trace, row=row, col=col)
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
    fig.update_layout(
        title_text="Communication Networks by Topic",
        height=350 * n_rows,
        width=1200,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# Helper to create PyVis HTML for a topic subgraph with centrality

def create_topic_pyvis_html(topic, centrality_measure, label_font_size=24):
    df = pd.read_csv('data/topic_model.csv')
    topic_df = df[df['topic'] == topic]
    G = nx.Graph()
    edge_weights = {}
    for _, row in topic_df.iterrows():
        src = row['source_entity']
        tgt = row['target_entity']
        if pd.notnull(src) and pd.notnull(tgt):
            if (src, tgt) not in edge_weights:
                edge_weights[(src, tgt)] = 0
            edge_weights[(src, tgt)] += 1
            G.add_edge(src, tgt)
    # Centrality
    if centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.degree_centrality(G)
    # PyVis
    net = Network(height="500px", width="100%", notebook=False, directed=False)
    # Scale node size by centrality (min size 10, max size 60)
    cent_vals = list(centrality.values())
    if cent_vals:
        min_cent, max_cent = min(cent_vals), max(cent_vals)
        def scale_size(val):
            if max_cent == min_cent:
                return 30  # fallback if all centralities are equal
            return 10 + 50 * (val - min_cent) / (max_cent - min_cent)
    else:
        def scale_size(val):
            return 30
    for node in G.nodes():
        cent_val = centrality.get(node, 0)
        net.add_node(
            node,
            label=node,
            size=scale_size(cent_val),
            title=f"{node}<br>{centrality_measure.capitalize()} Centrality: {cent_val:.3f}",
            color="#0074D9",
            font={"size": label_font_size, "color": "#111", "face": "Arial Black", "bold": True}
        )
    for (src, tgt), w in edge_weights.items():
        net.add_edge(src, tgt, width=1 + min(w, 10), color="#888888", arrows="to")
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.25, overlap=0.8)
    return net.generate_html()

# Dash layout additions
centrality_options = [
    {"label": "Betweenness Centrality", "value": "betweenness"},
    {"label": "Degree Centrality", "value": "degree"},
    {"label": "Closeness Centrality", "value": "closeness"},
]

topic_df = pd.read_csv('data/topic_model.csv')
topics = topic_df['topic'].dropna().unique().tolist()

def topic_pyvis_grid(centrality_measure):  
    n_cols = 3
    n_rows = (len(topics) + n_cols - 1) // n_cols
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx < len(topics):
                topic = topics[idx]
                html_str = create_topic_pyvis_html(topic, centrality_measure, label_font_size=28)
                row.append(html.Div([
                    html.Div(str(topic), style={"fontWeight": "bold", "fontSize": "1.5em", "textAlign": "center", "marginBottom": "8px"}),
                    html.Iframe(srcDoc=html_str, width="100%", height="540px", style={"border": "1px solid #ccc", "margin": "8px"})
                ], style={"flex": 1, "minWidth": "350px", "maxWidth": "600px", "margin": "0 8px"}))
            else:
                row.append(html.Div())
        grid.append(html.Div(row, style={"display": "flex", "flexDirection": "row", "justifyContent": "center", "marginBottom": "32px"}))
    return html.Div(grid)

# Get all unique entities for dropdown
entity_df = pd.read_csv('data/topic_model.csv')
all_entities = sorted(set(entity_df['source_entity'].dropna().unique()) | set(entity_df['target_entity'].dropna().unique()))
entity_options = [{"label": e, "value": e} for e in all_entities]
# For sunburst entity dropdown (same as entity_options)
sunburst_entity_options = entity_options.copy()

# Helper to create overall subgraph for selected entities

def create_entity_overall_pyvis_html(entities, centrality_measure, label_font_size=28):
    df = pd.read_csv('data/topic_model.csv')
    # Only keep rows where any selected entity is the source
    mask = df['source_entity'].isin(entities)
    sub_df = df[mask]
    G = nx.Graph()
    edge_weights = {}
    mentioned_entities = set()
    # Entity type color mapping (same as build_entity_network)
    entity_type_colors = {
        'Person': '#0074D9',      # Blue for people
        'Vessel': '#3D9970',      # Green for vessels
        'Location': '#FFDC00',    # Yellow for locations
        'Organization': '#FF4136', # Red for organizations
        'Other': '#AAAAAA',       # Gray for others
        'uncategorized': '#000000', # Black for uncategorized
        'mentioned_entity': '#000000' # Black for mentioned entities
    }
    # Build entity type lookup from all rows
    entity_types = {}
    for _, row in df.iterrows():
        src = row.get('source_entity')
        tgt = row.get('target_entity')
        src_type = row.get('source_entity_type', 'uncategorized')
        tgt_type = row.get('target_entity_type', 'uncategorized')
        if pd.notnull(src):
            entity_types[src] = src_type
        if pd.notnull(tgt):
            entity_types[tgt] = tgt_type
    for _, row in sub_df.iterrows():
        src = row['source_entity']
        tgt = row['target_entity']
        if pd.notnull(src) and pd.notnull(tgt):
            if (src, tgt) not in edge_weights:
                edge_weights[(src, tgt)] = 0
            edge_weights[(src, tgt)] += 1
            G.add_edge(src, tgt)
        # Add mentioned entities if present
        entities_str = str(row.get('Entities', ''))
        if entities_str.startswith('[') and entities_str.endswith(']'):
            entities_str = entities_str[1:-1]
            mentioned = [e.strip().strip("'\"") for e in entities_str.split(',') if e.strip()]
        else:
            mentioned = [e.strip() for e in entities_str.split(',') if e.strip()]
        for ent in mentioned:
            if ent:
                mentioned_entities.add(ent)
                G.add_edge(src, ent)
                if ent not in entity_types:
                    entity_types[ent] = 'mentioned_entity'
    # Centrality
    if centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.degree_centrality(G)
    net = Network(height="500px", width="100%", notebook=False, directed=False)
    for node in G.nodes():
        cent_val = centrality.get(node, 0)
        highlight = node in entities
        is_mentioned = node in mentioned_entities
        node_type = entity_types.get(node, 'uncategorized')
        color = entity_type_colors.get(node_type, '#000000')
        if highlight:
            color = '#FF4136'  # Always highlight selected entities in red
        net.add_node(
            node,
            label=node,
            size=24 + 50 * cent_val if highlight else (18 + 40 * cent_val if not is_mentioned else 16 + 30 * cent_val),
            title=f"{node} ({node_type})<br>{centrality_measure.capitalize()} Centrality: {cent_val:.3f}",
            color=color,
            font={"size": label_font_size, "color": "#111", "face": "Arial Black", "bold": True}
        )
    for (src, tgt), w in edge_weights.items():
        net.add_edge(src, tgt, width=1 + min(w, 10), color="#888888", arrows="to")
    # Add edges from source to mentioned entities (if not already in edge_weights)
    for _, row in sub_df.iterrows():
        src = row['source_entity']
        entities_str = str(row.get('Entities', ''))
        if entities_str.startswith('[') and entities_str.endswith(']'):
            entities_str = entities_str[1:-1]
            mentioned = [e.strip().strip("'\"") for e in entities_str.split(',') if e.strip()]
        else:
            mentioned = [e.strip() for e in entities_str.split(',') if e.strip()]
        for ent in mentioned:
            if ent and (src, ent) not in edge_weights:
                net.add_edge(src, ent, width=1, color="#bbbbbb", dashes=True)
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.25, overlap=0.8)
    return net.generate_html()

# Helper to create topic subgraphs for selected entities

def create_entity_topic_pyvis_html(entities, topic, centrality_measure, label_font_size=28):
    df = pd.read_csv('data/topic_model.csv')
    topic_df = df[df['topic'] == topic]
    # Only keep rows where any selected entity is source or target
    mask = topic_df['source_entity'].isin(entities) | topic_df['target_entity'].isin(entities)
    topic_df = topic_df[mask]
    G = nx.Graph()
    edge_weights = {}
    for _, row in topic_df.iterrows():
        src = row['source_entity']
        tgt = row['target_entity']
        if pd.notnull(src) and pd.notnull(tgt):
            if (src, tgt) not in edge_weights:
                edge_weights[(src, tgt)] = 0
            edge_weights[(src, tgt)] += 1
            G.add_edge(src, tgt)
    if len(G) == 0:
        return None
    if centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.degree_centrality(G)
    net = Network(height="500px", width="100%", notebook=False, directed=False)
    for node in G.nodes():
        cent_val = centrality.get(node, 0)
        highlight = node in entities
        net.add_node(
            node,
            label=node,
            size=24 + 50 * cent_val if highlight else 18 + 40 * cent_val,
            title=f"{node}<br>{centrality_measure.capitalize()} Centrality: {cent_val:.3f}",
            color="#FF4136" if highlight else "#0074D9",
            font={"size": label_font_size, "color": "#111", "face": "Arial Black", "bold": True}
        )
    for (src, tgt), w in edge_weights.items():
        net.add_edge(src, tgt, width=1 + min(w, 10), color="#888888", arrows="to")
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.25, overlap=0.8)
    return net.generate_html()

# Entity filter and subgraph grid

def entity_subgraph_section(centrality_measure, selected_entities):
    if not selected_entities:
        return html.Div()
    # Overall subgraph
    overall_html = create_entity_overall_pyvis_html(selected_entities, centrality_measure)
    overall = html.Div([
        html.H4("Overall Communication Subgraph", style={"textAlign": "center", "marginTop": "24px"}),
        html.Iframe(srcDoc=overall_html, width="100%", height="540px", style={"border": "1px solid #ccc", "margin": "8px"})
    ])
    # Topic subgraphs
    df = pd.read_csv('data/topic_model.csv')
    involved_topics = df[(df['source_entity'].isin(selected_entities)) | (df['target_entity'].isin(selected_entities))]['topic'].dropna().unique().tolist()
    topic_grids = []
    for topic in involved_topics:
        html_str = create_entity_topic_pyvis_html(selected_entities, topic, centrality_measure)
        if html_str:
            topic_grids.append(html.Div([
                html.Div(str(topic), style={"fontWeight": "bold", "fontSize": "1.5em", "textAlign": "center", "marginBottom": "8px"}),
                html.Iframe(srcDoc=html_str, width="100%", height="540px", style={"border": "1px solid #ccc", "margin": "8px"})
            ], style={"flex": 1, "minWidth": "350px", "maxWidth": "600px", "margin": "0 8px", "marginBottom": "32px"}))
    return html.Div([
        overall,
        html.H4("Topic Subgraphs for Selected Entities", style={"textAlign": "center", "marginTop": "32px"}),
        html.Div(topic_grids, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"})
    ])

# =====================
# Motif Extraction and Sankey Prep for Variable-Length Motifs
# =====================
def extract_repeating_motifs_all_lengths(df, min_days=2, min_count=2, max_length=8):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%y %H:%M')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')
    motif_counts = defaultdict(int)
    motif_days = defaultdict(set)
    for date, group in df.groupby('date'):
        group = group.sort_values('timestamp')
        pairs = list(zip(group['source'], group['target']))
        n = len(pairs)
        for L in range(2, max_length+1):
            for i in range(n - L + 1):
                # Check for a valid chain: target of one is source of next
                valid_chain = True
                for j in range(L-1):
                    if pairs[i+j][1] != pairs[i+j+1][0]:
                        valid_chain = False
                        break
                if not valid_chain:
                    continue
                # Build motif: first source, then all intermediate targets, then last target
                motif = [pairs[i][0]]
                for j in range(L-1):
                    motif.append(pairs[i+j][1])
                motif = tuple(motif)
                motif_counts[motif] += 1
                motif_days[motif].add(date)
    repeating_motifs = {m: motif_days[m] for m in motif_counts if len(motif_days[m]) >= min_days and motif_counts[m] >= min_count}
    return repeating_motifs, motif_counts

def prepare_sankey_data_varlen(repeating_motifs, motif_counts, top_n=10):
    top_motifs = sorted(repeating_motifs, key=lambda m: motif_counts[m], reverse=True)[:top_n]
    labels = []
    label_map = {}
    idx = 0
    sources = []
    targets = []
    values = []
    for motif in top_motifs:
        for node in motif:
            if node not in label_map:
                label_map[node] = idx
                labels.append(node)
                idx += 1
        for i in range(len(motif) - 1):
            sources.append(label_map[motif[i]])
            targets.append(label_map[motif[i+1]])
            values.append(motif_counts[motif])
    return labels, sources, targets, values, top_motifs

def create_motif_sankey(labels, sources, targets, values):
    import plotly.graph_objects as go

    # Identify source and sink nodes
    source_set = set(sources)
    target_set = set(targets)
    node_types = []
    node_x = []
    for i, label in enumerate(labels):
        if i in source_set and i not in target_set:
            node_types.append('source')
            node_x.append(0.0)
        elif i in target_set and i not in source_set:
            node_types.append('sink')
            node_x.append(1.0)
        else:
            node_types.append('intermediate')
            node_x.append(0.5)

    # Assign colors
    color_map = {
        'source': '#4F8EF7',        # Blue for sources
        'sink': '#F76F4F',          # Red for sinks
        'intermediate': '#B0B0B0'   # Gray for intermediates
    }
    node_colors = [color_map[t] for t in node_types]

    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            pad=80,
            thickness=12,
            line=dict(color="black", width=0.5),
            color=node_colors,
            x=node_x,  # Set node positions for discrete separation
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(160,160,160,0.30)"
        )
    ))
    fig.update_layout(
        title_text="Repeating communication patterns",
        font_size=18,
        height=1200,
        width=1800,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    return fig

def plot_motif_network(motif, count):
    """
    motif: tuple of node names (e.g., ('A', 'B', 'A'))
    count: int, how many times this motif occurs
    Returns: Plotly Figure
    """
    import networkx as nx
    import plotly.graph_objects as go
    G = nx.DiGraph()
    # Add edges for the motif sequence
    for i in range(len(motif) - 1):
        G.add_edge(motif[i], motif[i+1], weight=count)
    # Optionally, add edge from last to first if it's a loop
    if motif[0] == motif[-1] and len(motif) > 2:
        G.add_edge(motif[-2], motif[-1], weight=count)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    edge_x, edge_y = [], []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            color='#4F8EF7',
            size=30,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=f"Motif: {' → '.join(motif)} (Count: {count})",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        width=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def plot_motifs_network(motifs, counts, top_n=10):
    """
    motifs: dict of motif tuple -> set of days
    counts: dict of motif tuple -> count
    top_n: number of top motifs to show
    Returns: Plotly Figure
    """
    import networkx as nx
    import plotly.graph_objects as go
    # Get top N motifs by count
    sorted_motifs = sorted(counts, key=counts.get, reverse=True)[:top_n]
    G = nx.DiGraph()
    # Aggregate all unique edges
    for motif in sorted_motifs:
        for i in range(len(motif) - 1):
            src, tgt = motif[i], motif[i+1]
            G.add_edge(src, tgt)
    # Find all weakly connected components
    components = list(nx.weakly_connected_components(G))
    node_traces = []
    edge_traces = []
    arrow_traces = []
    x_offset = 0
    x_gap = 3.5  # horizontal gap between subgraphs
    for comp in components:
        subG = G.subgraph(comp).copy()
        # Layout for this subgraph
        pos = nx.spring_layout(subG, seed=42, k=1.5)
        # Offset positions to avoid overlap between subgraphs
        pos = {n: (x + x_offset, y) for n, (x, y) in pos.items()}
        # Edges and arrows
        for src, tgt in subG.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=2, color='#444'),
                hoverinfo='text',
                mode='lines',
                text=[f"{src} → {tgt}"]
            ))
            # Arrow for direction
            arrow_x = x0 * 0.7 + x1 * 0.3
            arrow_y = y0 * 0.7 + y1 * 0.3
            arrow_traces.append(go.Scatter(
                x=[arrow_x, x1],
                y=[arrow_y, y1],
                mode='lines+markers',
                line=dict(color='#222', width=2),
                marker=dict(symbol='arrow', size=18, angleref='previous', color='#222'),
                showlegend=False,
                hoverinfo='skip'
            ))
        # Nodes
        node_x, node_y, node_text = [], [], []
        for node in subG.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        node_traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                color='#4F8EF7',
                size=30,
                line=dict(width=2, color='black')
            ),
            hoverinfo='text'
        ))
        # Offset for next subgraph
        x_offset += x_gap
    fig = go.Figure(edge_traces + arrow_traces + node_traces)
    fig.update_layout(
        title=f"Motif Communication Components as Directed Network Graph",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        height=700,
        width=1200,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# =====================
# Dash Layout - Add Motif Sankey to Temporal Patterns Tab
# =====================
# Extract motifs and prepare Sankey data (do this once at startup for performance)
repeating_motifs, motif_counts = extract_repeating_motifs_all_lengths(comms_df, min_days=2, min_count=2, max_length=4)
labels_motif, sources_motif, targets_motif, values_motif, top_motifs = prepare_sankey_data_varlen(repeating_motifs, motif_counts, top_n=len(repeating_motifs))
fig_motif_sankey = create_motif_sankey(labels_motif, sources_motif, targets_motif, values_motif)

# Store the full motif extraction for all lengths (2 to 8)
repeating_motifs_all, motif_counts_all = extract_repeating_motifs_all_lengths(comms_df, min_days=2, min_count=2, max_length=8)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "DOCK-G: Decoding Oceanus's Corruption Knowledge-Graphs"
server = app.server


# Simple Chatbot UI
chatbot_card = dbc.Card([
    dbc.CardHeader([
        html.H4("Ask DOCK-G!!", style={"marginBottom": "0"}),
        html.Div([
            html.Span("💬 Session: ", style={"fontSize": "0.9em", "color": "#666"}),
            html.Span(id="chatbot-session-counter", children="0/5", style={"fontSize": "0.9em", "fontWeight": "bold", "color": "#0074D9"})
        ], style={"marginTop": "8px"})
    ]),
    dbc.CardBody([
        dcc.Store(id="chatbot-history", data=[]),
        # Session warning message
        html.Div(
            id="chatbot-session-warning",
            children="⚠️ **Session Info**: Your conversation will reset after 5 messages to maintain optimal performance.",
            style={
                "background": "#fff3cd",
                "border": "1px solid #ffeaa7",
                "borderRadius": "8px",
                "padding": "12px",
                "marginBottom": "12px",
                "fontSize": "0.9em",
                "color": "#856404"
            }
        ),
        html.Div(
            id="chatbot-chat-area",
            style={
                "height": "300px",
                "maxHeight": "35vh",
                "overflowY": "auto",
                "background": "#f7f7fa",
                "borderRadius": "12px",
                "padding": "18px 12px",
                "marginBottom": "16px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.04)",
                "display": "flex",
                "flexDirection": "column",
                "gap": "12px"
            }
        ),
        dbc.Row([
            dbc.Col([
                dcc.Input(
                    id="chatbot-input",
                    type="text",
                    placeholder="Type your question here...",
                    style={"width": "100%", "borderRadius": "8px", "padding": "10px", "fontSize": "1.1em"},
                    debounce=True,
                    autoFocus=True
                )
            ], width=9),
            dbc.Col([
                dbc.Button("Send", id="chatbot-send", color="primary", n_clicks=0, style={"width": "100%", "fontSize": "1.1em", "borderRadius": "8px"})
            ], width=3)
        ], className="g-1"),
        # Loading spinner
        html.Div(
            id="chatbot-loading",
            children=[
                dbc.Spinner(
                    size="sm",
                    color="primary",
                    type="border"
                ),
                html.Span(" Processing your question...", style={"marginLeft": "8px", "fontSize": "0.9em", "color": "#666"})
            ],
            style={
                "display": "none",
                "justifyContent": "center",
                "alignItems": "center",
                "padding": "12px",
                "background": "#f8f9fa",
                "borderRadius": "8px",
                "marginTop": "8px",
                "border": "1px solid #e9ecef"
            }
        ),
        # Hidden interval for loading management
        dcc.Interval(
            id="chatbot-loading-interval",
            interval=100,  # 100ms
            n_intervals=0,
            disabled=True
        )
    ])
], style={"marginBottom": "32px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})

# Helper to build motif details table
def build_motif_details_table(filtered_motifs, min_length, min_days):
    # Load the original comms_df for message lookup
    df = comms_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%y %H:%M')
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')
    rows = []
    for motif, days in filtered_motifs.items():
        # For each day, find all matching chains/messages
        for day in sorted(days):
            day_df = df[df['date'] == day].sort_values('timestamp')
            pairs = list(zip(day_df['source'], day_df['target']))
            n = len(pairs)
            for i in range(n - len(motif) + 1):
                # Check if this chain matches the motif
                chain = [pairs[i][0]]
                for j in range(len(motif)-1):
                    chain.append(pairs[i+j][1])
                if tuple(chain) == motif:
                    # Collect the messages and timestamps for this motif occurrence
                    msg_indices = list(range(i, i+len(motif)-1+1))
                    for idx in msg_indices:
                        row = day_df.iloc[idx]
                        rows.append({
                            'motif': ' → '.join(motif),
                            'day': day,
                            'timestamp': row['timestamp'],
                            'source': row['source'],
                            'target': row['target'],
                            'Message_ID': row.get('Message_ID', ''),
                            'Processed_Text': row.get('Processed_Text', '')
                        })
    # Sort by timestamp
    rows = sorted(rows, key=lambda r: r['timestamp'])
    # Remove duplicate rows (same motif, day, timestamp, source, target)
    seen = set()
    unique_rows = []
    for r in rows:
        key = (r['motif'], r['day'], r['timestamp'], r['source'], r['target'])
        if key not in seen:
            unique_rows.append(r)
            seen.add(key)
    columns = [
        {'name': 'Motif', 'id': 'motif'},
        {'name': 'Day', 'id': 'day'},
        {'name': 'Timestamp', 'id': 'timestamp'},
        {'name': 'Source', 'id': 'source'},
        {'name': 'Target', 'id': 'target'},
        {'name': 'Message_ID', 'id': 'Message_ID'},
        {'name': 'Message', 'id': 'Processed_Text'},
    ]
    return dash_table.DataTable(
        columns=columns,
        data=unique_rows,
        style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'left', 'fontSize': '1em', 'minWidth': '80px', 'maxWidth': '400px', 'whiteSpace': 'normal'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f7f7fa'},
        page_size=20,
        sort_action='native',
        filter_action='native',
        export_format='csv',
    )

# Update Temporal Patterns Tab Layout to add dropdown

def motif_tab_layout():
    # --- Prepare unique dates for slider ---
    comms_df['parsed_date'] = pd.to_datetime(comms_df['date'], format='%d/%m/%y')
    unique_dates = sorted(comms_df['parsed_date'].unique())
    day_marks = {i+1: f"Day {i+1}" for i in range(len(unique_dates))}

    return html.Div([
        # --- Sunburst Title ---
        html.H3("Daily communications across groups of entities", style={"textAlign": "center", "marginTop": "32px", "marginBottom": "16px"}),
        # --- Show Explanation Button and Collapse ---
        dbc.Button(
            "Show Explanation",
            id="collapse-button",
            className="mb-2",
            color="info",
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Alert(
                [
                    html.H5("What is a Packet?", className="alert-heading"),
                    html.P(
                        "A packet is defined as any sequence of messages where a new packet is begun if the first message of the (N+1)st packet is more than 15 minutes after the last message of packet N."
                    ),
                    html.H5("How are 'Pairs' Created?", className="alert-heading mt-3"),
                    html.P(
                        "'Pairs' represent unique sender-receiver combinations for each message or communication event within a packet. For example, if 'the lookout' sends a message to 'the intern', the pair is 'the lookout <-> the intern'. If a message is sent in one direction only, it is shown as 'sender -> receiver'."
                    ),
                ],
                color="info",
                className="mb-4",
                style={"fontSize": "1.1em"}
            ),
            id="collapse-explainer",
            is_open=False,
        ),
        # --- Entity Filter Dropdown ---
        html.Div([
            html.Label("Highlight Entities in Sunburst:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="sunburst-entity-dropdown",
                options=sunburst_entity_options,
                multi=True,
                value=[],
                style={"width": "600px", "marginBottom": "24px"}
            )
        ], style={"marginBottom": "8px", "width": "100%"}),
        dcc.Graph(
            id="sunburst-temporal-patterns",
            style={"width": "100%", "height": "900px"}
        ),
        # --- NEW: Packet Candlestick/Network/Selected Messages Section ---
        html.Hr(),
        html.H3("Packet Communication Volume and Analysis", style={"textAlign": "center", "marginTop": "32px", "marginBottom": "16px"}),
        html.Div([
            html.Label("Select Day:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Slider(
                id="packet-day-slider",
                min=1,
                max=len(unique_dates),
                step=1,
                value=1,
                marks=day_marks,
                included=False
            ),
            html.Label("Time Interval (minutes):", style={"fontWeight": "bold", "marginTop": "18px", "marginRight": "10px"}),
            dcc.Slider(
                id="packet-interval-slider",
                min=1,
                max=60,
                step=1,
                value=1,  # <-- Set default to 1 minute
                marks={1: "1m", 5: "5m", 10: "10m", 15: "15m", 30: "30m", 60: "60m"},
                included=False,
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={"marginBottom": "20px", "display": "block", "width": "100%"}),
        dcc.Graph(id="packet-candlestick-chart", style={"height": "400px"}),
        html.Div([
            html.H4("Packet Network Analysis", style={"marginTop": "20px", "marginBottom": "10px"}),
            html.P("Click on any triangle in the chart above to view the network analysis for that packet."),
            html.Div(id="packet-network-pane", style={"marginTop": "10px"})
        ]),
        html.Div([
            html.H4("Selected Messages", style={"marginTop": "20px", "marginBottom": "10px"}),
            html.P("Messages highlighted in yellow are part of potential pseudonym groups (similar message pairs).", 
                   style={"fontSize": "12px", "color": "#666", "marginBottom": "10px", "fontStyle": "italic"}),
            dash_table.DataTable(
                id="packet-messages-table",
                columns=[
                    {"name": "Message ID", "id": "Message_ID"},
                    {"name": "Source", "id": "source"},
                    {"name": "Target", "id": "target"},
                    {"name": "Content", "id": "content"}
                ],
                style_table={"height": "400px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"fontWeight": "bold"},
                page_size=20,
                style_data_conditional=[],
                style_cell_conditional=[
                    {"if": {"column_id": "content"}, "whiteSpace": "pre-line"}
                ]
            )
        ]),
        html.Hr(),
        # --- Motif Length Slider and Days Dropdown ---
        html.H3("Repeating communication patterns", style={"textAlign": "center", "marginTop": "32px", "marginBottom": "16px"}),
        html.Div([
            html.Div([
                html.Label("Minimum Motif Length:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Slider(
                    id="motif-length-slider",
                    min=2,
                    max=8,
                    step=1,
                    value=2,
                    marks={i: str(i) for i in range(2, 9)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    included=False,
                    updatemode="drag"
                )
            ], style={"marginBottom": "24px", "width": "60%"}),
            html.Div([
                html.Label("Minimum Days Motif Repeats:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="motif-days-dropdown",
                    options=[{"label": str(i), "value": i} for i in range(2, 15)],
                    value=2,
                    clearable=False,
                    style={"width": "120px", "display": "inline-block"}
                )
            ], style={"marginBottom": "8px", "width": "60%"})
        ], style={"marginTop": "32px", "marginBottom": "8px", "width": "100%"}),
        # --- Motif Sankey Visual ---
        html.Iframe(
            id="motif-network-iframe",
            srcDoc="",
            width="100%",
            height="700px",
            style={"border": "1px solid #ccc", "marginTop": "20px"}
        ),
        html.Hr(),
        # --- Windrose Entity Dropdown ---
        html.Div([
            html.H3("Visualize influence on entity", style={"textAlign": "center", "marginTop": "32px", "marginBottom": "16px"}),
            html.Label("Select Entity for Wind Rose:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="windrose-entity-dropdown",
                options=entity_options,
                value="mako",
                clearable=False,
                style={"width": "400px", "marginBottom": "24px"}
            )
        ], style={"marginBottom": "8px", "width": "100%"}),
        dcc.Graph(id="mako-windrose-graph", style={"width": "100%", "height": "800px", "margin": "0 auto"}),
    ], className="p-4")

# --- Helper for packet candlestick chart, network, and messages ---
def build_packet_candlestick_figure(selected_day_idx, interval_minutes):
    # Get unique dates
    comms_df['parsed_date'] = pd.to_datetime(comms_df['date'], format='%d/%m/%y')
    unique_dates = sorted(comms_df['parsed_date'].unique())
    if selected_day_idx < 1 or selected_day_idx > len(unique_dates):
        return go.Figure()
    day = unique_dates[selected_day_idx - 1]
    filtered = comms_df[comms_df['parsed_date'] == day].copy()
    filtered['timestamp'] = pd.to_datetime(filtered['timestamp'], format='%d/%m/%y %H:%M')
    # Filter to business hours (8am to 2:30pm)
    filtered = filtered[(filtered['hour'] >= 8) & (filtered['hour'] <= 14)]
    filtered = filtered[~((filtered['hour'] == 14) & (filtered['timestamp'].dt.minute > 30))]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Communication Volume on {day.strftime('%Y-%m-%d')} (8am-2:30pm, {interval_minutes}min intervals) - No Data",
            xaxis_title="Time",
            yaxis_title="Message Count",
            height=400
        )
        return fig
    # Create time intervals
    start_time = filtered['timestamp'].min().replace(hour=8, minute=0, second=0, microsecond=0)
    end_time = filtered['timestamp'].max().replace(hour=14, minute=30, second=0, microsecond=0)
    intervals = []
    current_time = start_time
    while current_time <= end_time:
        interval_end = current_time + pd.Timedelta(minutes=interval_minutes)
        intervals.append((current_time, interval_end))
        current_time = interval_end
    interval_data = []
    dormant_intervals = []
    for interval_start, interval_end in intervals:
        interval_messages = filtered[(filtered['timestamp'] >= interval_start) & (filtered['timestamp'] < interval_end)]
        count = len(interval_messages)
        if count > 0:
            packet_counts = interval_messages.groupby('packet_id').size().reset_index(name='count')
            for _, row in packet_counts.iterrows():
                packet_id = row['packet_id']
                packet_count = row['count']
                interval_data.append({
                    'time': interval_start,
                    'count': packet_count,
                    'packet_id': packet_id,
                    'total_count': count
                })
        else:
            dormant_intervals.append(interval_start)
    interval_df = pd.DataFrame(interval_data)
    fig = go.Figure()
    unique_packets = interval_df['packet_id'].unique()
    colors = plotly.colors.qualitative.Plotly
    if not interval_df.empty:
        for i, packet_id in enumerate(unique_packets):
            packet_data = interval_df[interval_df['packet_id'] == packet_id]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=packet_data['time'],
                y=packet_data['count'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color=color,
                    line=dict(color='black', width=1)
                ),
                name=f'Packet {packet_id}',
                hovertemplate='<b>Packet %{customdata}</b><br>' +
                              'Time: %{x}<br>' +
                              'Messages: %{y}<br>' +
                              '<extra></extra>',
                customdata=[packet_id] * len(packet_data),
                showlegend=True
            ))
    if dormant_intervals:
        fig.add_trace(go.Scatter(
            x=dormant_intervals,
            y=[0] * len(dormant_intervals),
            mode='markers',
            marker=dict(
                symbol='circle',
                size=10,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            name='Dormant Intervals',
            showlegend=True
        ))
    fig.update_layout(
        title=f"Communication Volume on {day.strftime('%Y-%m-%d')} (8am-2:30pm, {interval_minutes}min intervals) - Colored by Packet ID",
        xaxis_title="Time",
        yaxis_title="Message Count",
        xaxis=dict(type='date', tickformat='%H:%M', dtick=5*60*1000, tickmode='linear'),
        yaxis=dict(title="Volume (Message Count)", rangemode='tozero', dtick=1, tickmode='linear'),
        height=400,
        xaxis_rangeslider=dict(visible=True)
    )
    return fig

# --- Helper for packet network graph ---
def build_packet_network_pyvis_html(packet_id):
    df = comms_df[comms_df['packet_id'] == packet_id]
    if df.empty:
        return "<html><body><h4>No data for this packet.</h4></body></html>"
    import networkx as nx
    from pyvis.network import Network
    G = nx.DiGraph()
    edge_weights = {}
    for _, row in df.iterrows():
        src, tgt = row['source'], row['target']
        G.add_edge(src, tgt)
        key = (src, tgt)
        edge_weights[key] = edge_weights.get(key, 0) + 1
    # Centrality measures
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = nx.degree_centrality(G)
    net = Network(height="500px", width="100%", notebook=False, directed=True)
    for node in G.nodes():
        net.add_node(
            node,
            label=node,
            size=24,
            color="#4F8EF7",
            font={"size": 22, "color": "#111", "face": "Arial Black", "bold": True},
            title=(
                f"<b>{node}</b><br>"
                f"Betweenness: {betweenness[node]:.3f}<br>"
                f"Closeness: {closeness[node]:.3f}<br>"
                f"Degree: {degree[node]:.3f}"
            )
        )
    for src, tgt in G.edges():
        key = (src, tgt)
        weight = edge_weights.get(key, 1)
        net.add_edge(
            src, tgt,
            width=1 + 3 * min(weight, 10),
            color="#888888",
            title=f"{src} → {tgt}: {weight} messages",
            arrows="to"
        )
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.25, overlap=0.8)
    return net.generate_html()

# --- Callback for packet section ---
from dash.dependencies import Input, Output, State
@app.callback(
    [
        Output("packet-candlestick-chart", "figure"),
        Output("packet-network-pane", "children"),
        Output("packet-messages-table", "data"),
        Output("packet-messages-table", "style_data_conditional")
    ],
    [
        Input("packet-day-slider", "value"),
        Input("packet-interval-slider", "value"),
        Input("packet-candlestick-chart", "clickData")
    ]
)
def update_packet_section(selected_day, interval_minutes, click_data):
    fig = build_packet_candlestick_figure(selected_day, interval_minutes)
    network_pane = None
    messages_data = []
    style_data_conditional = []
    
    # If a triangle (packet) is clicked, show network and messages for that packet
    if click_data and 'points' in click_data and len(click_data['points']) > 0:
        clicked_point = click_data['points'][0]
        if 'customdata' in clicked_point and clicked_point['customdata']:
            packet_id = clicked_point['customdata']
            pyvis_html = build_packet_network_pyvis_html(packet_id)
            network_pane = html.Iframe(srcDoc=pyvis_html, width="100%", height="540px", style={"border": "1px solid #ccc", "margin": "8px"})
            # Show all messages for this packet
            df = comms_df[comms_df['packet_id'] == packet_id]
            messages_data = df[['Message_ID', 'source', 'target', 'content']].to_dict('records')
            
            # Check for pseudonym matches and add highlighting
            if not df.empty:
                # Get all Message_IDs in this packet
                packet_message_ids = set(df['Message_ID'].astype(int))
                
                # Find matching rows in pseudonyms_df
                matching_source_ids = set(pseduonyms_df['source_id'].astype(int))
                matching_target_ids = set(pseduonyms_df['target_id'].astype(int))
                
                # Find Message_IDs that appear in pseudonyms_df
                pseudonym_message_ids = packet_message_ids.intersection(matching_source_ids.union(matching_target_ids))
                
                # Create a mapping of Message_ID to pseudonym groups (can have multiple)
                msg_to_pseudonym_groups = {}
                for _, row in pseduonyms_df.iterrows():
                    source_id = int(row['source_id'])
                    target_id = int(row['target_id'])
                    pseudonym_id = row['potential_pseduonym_id']
                    
                    if source_id in packet_message_ids:
                        if source_id not in msg_to_pseudonym_groups:
                            msg_to_pseudonym_groups[source_id] = set()
                        msg_to_pseudonym_groups[source_id].add(pseudonym_id)
                    if target_id in packet_message_ids:
                        if target_id not in msg_to_pseudonym_groups:
                            msg_to_pseudonym_groups[target_id] = set()
                        msg_to_pseudonym_groups[target_id].add(pseudonym_id)
                
                # Modify messages_data to add pseudonym group info
                for msg in messages_data:
                    msg_id = int(msg['Message_ID'])
                    if msg_id in msg_to_pseudonym_groups:
                        pseudonym_groups = sorted(msg_to_pseudonym_groups[msg_id])
                        pseudonym_text = ", ".join([f"#{group}" for group in pseudonym_groups])
                        msg['Message_ID'] = f"{msg_id} ({pseudonym_text})"
                
                # Add highlighting for matching rows
                for msg_id in pseudonym_message_ids:
                    style_data_conditional.append({
                        'if': {'filter_query': f'{{Message_ID}} contains "{msg_id} ("'},
                        'backgroundColor': '#fff3cd',  # Light yellow background
                        'border': '2px solid #ffc107',  # Yellow border
                        'fontWeight': 'bold'
                    })
    
    return fig, network_pane, messages_data, style_data_conditional

# =====================
# Mako Wind Rose Visualization
# =====================

def build_mako_windrose_figure(selected_entity="mako"):
    import plotly.graph_objects as go
    import numpy as np
    df = comms_df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')
    entity = selected_entity or "mako"
    day_dates = sorted(df['date'].unique())
    n_days = len(day_dates)
    # Aggregate incoming and outgoing counts per entity per day
    incoming = df[df['target'] == entity].groupby(['source', 'date']).size().reset_index(name='in_count')
    outgoing = df[df['source'] == entity].groupby(['target', 'date']).size().reset_index(name='out_count')
    # Get all entities that interact with selected entity
    all_entities_set = set(incoming['source']).union(set(outgoing['target']))
    # Calculate total messages (incoming + outgoing) for each entity
    entity_total_msgs = {}
    for e in all_entities_set:
        in_count = incoming[incoming['source'] == e]['in_count'].sum()
        out_count = outgoing[outgoing['target'] == e]['out_count'].sum()
        entity_total_msgs[e] = in_count + out_count
    # Sort entities by total messages, descending
    all_entities = [e for e, _ in sorted(entity_total_msgs.items(), key=lambda x: x[1], reverse=True)]
    n_entities = len(all_entities)
    # Assign angles
    angles = np.linspace(0, 2*np.pi, num=n_entities, endpoint=False)
    entity_angle = {e: angles[i] for i, e in enumerate(all_entities)}
    # Create display labels with total interactions
    entity_display_labels = [f"{e} [{entity_total_msgs[e]}]" for e in all_entities]
    entity_display_label_map = {e: label for e, label in zip(all_entities, entity_display_labels)}
    # Prepare data for each (entity, day)
    points = []  # (r, theta, size, color, legend, entity, day, in_count, out_count)
    for i, e in enumerate(all_entities):
        theta = np.degrees(entity_angle[e])
        for day_idx, day in enumerate(day_dates):
            in_count = incoming[(incoming['source'] == e) & (incoming['date'] == day)]['in_count'].sum()
            out_count = outgoing[(outgoing['target'] == e) & (outgoing['date'] == day)]['out_count'].sum()
            if in_count == 0 and out_count == 0:
                continue
            if in_count > 0 and out_count > 0:
                color = 'red'
                legend = 'Both (in & out)'
            elif in_count > 0:
                color = 'royalblue'
                legend = 'Incoming only'
            else:
                color = 'gold'
                legend = 'Outgoing only'
            size = 8 + 6 * np.log1p(in_count + out_count)
            points.append(dict(r=day_idx+1, theta=theta, size=size, color=color, legend=legend, entity=e, day=day, in_count=in_count, out_count=out_count))
    # Group points by legend for separate traces
    traces = []
    for legend, color in [('Incoming only', 'royalblue'), ('Outgoing only', 'gold'), ('Both (in & out)', 'red')]:
        group = [p for p in points if p['legend'] == legend]
        if group:
            traces.append(go.Scatterpolar(
                r=[p['r'] for p in group],
                theta=[p['theta'] for p in group],
                mode='markers',
                marker=dict(size=[p['size'] for p in group], color=color, opacity=0.8, line=dict(width=1, color='black')),
                name=legend,
                hovertemplate="Entity: %{customdata[0]}<br>Day: %{customdata[1]}<br>Incoming: %{customdata[2]}<br>Outgoing: %{customdata[3]}<br>Total: %{customdata[4]}<extra></extra>",
                customdata=[[f"{entity_display_label_map[p['entity']]}", p['day'], p['in_count'], p['out_count'], p['in_count']+p['out_count']] for p in group]
            ))
    # Layout
    fig = go.Figure(traces)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                tickvals=list(range(1, n_days+1)),
                ticktext=["" for _ in range(1, n_days+1)],
                angle=90,
                dtick=1,
                showline=True,
                linewidth=2,
                gridcolor='#888',
                gridwidth=1
            ),
            angularaxis=dict(
                tickvals=[np.degrees(entity_angle[e]) for e in all_entities],
                ticktext=entity_display_labels,
                direction='clockwise',
                rotation=90
            ),
            bgcolor='#222'
        ),
        showlegend=True,
        legend=dict(title='Direction', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        title="Entity Communication Patterns over 14 days",
        height=800,
        width=900,
        paper_bgcolor='#222',
        font=dict(color='white')
    )
    return fig

# =====================
# Nadia Conti Tab Layout
# =====================
def create_nadia_conti_tab_layout():
    """Create the layout for the Nadia Conti tab (checkbox always visible above network analysis)"""
    return html.Div([
        html.H3("Nadia Conti Communication Timeline"),
        html.P("This Gantt chart shows all communications involving Nadia Conti as a source, target, or mentioned entity."),
        html.P("Click on a packet in the timeline below to view its subgraph", style={"color": "#0074D9", "fontWeight": "bold", "marginBottom": "18px"}),
        dcc.Loading(
            id="loading-nadia-gantt",
            children=dcc.Graph(id="nadia-gantt-chart"),
            type="default"
        ),
        html.Div(
            id="llm-summary-container",
            style={"marginTop": "20px", "padding": "10px", "borderRadius": "5px"},
            children=[
                dcc.Checklist(
                    id="llm-summary-checkbox",
                    options=[{"label": "Generate LLM Summary", "value": "generate"}],
                    value=[],
                    style={"marginTop": "10px", "marginBottom": "10px"}
                ),
                dcc.Loading(
                    id="loading-llm-summary",
                    children=html.Div(id="nadia-packet-summary"),
                    type="default"
                )
            ]
        ),
    ], style={"margin": "30px"})

# =====================
# Nadia Conti Gantt Chart Helper
# =====================
def build_nadia_gantt_figure():
    """
    Build Gantt chart showing all communications involving Nadia Conti
    as source, target, or mentioned entity
    """
    # Filter communications involving Nadia Conti or "boss"
    nadia_communications = []
    for idx, row in comms_df.iterrows():
        # Check if Nadia is source or target (case insensitive)
        source_lower = str(row['source']).lower()
        target_lower = str(row['target']).lower()
        is_source = 'nadia' in source_lower or 'boss' in source_lower
        is_target = 'nadia' in target_lower or 'boss' in target_lower
        # Check if Nadia is mentioned in entities
        is_mentioned = False
        try:
            entities = eval(row['Entities']) if isinstance(row['Entities'], str) else row['Entities']
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, str) and ('nadia' in ent.lower() or 'boss' in ent.lower()):
                        is_mentioned = True
                        break
        except Exception:
            pass
        if is_source or is_target or is_mentioned:
            # Determine role combination
            roles = []
            if is_source:
                roles.append('Source')
            if is_target:
                roles.append('Target')
            if is_mentioned:
                roles.append('Mentioned')
            role_combination = ' + '.join(sorted(roles))
            nadia_communications.append({
                'timestamp': pd.to_datetime(row['timestamp'], format='%d/%m/%y %H:%M'),
                'date': pd.to_datetime(row['date'], format='%d/%m/%y'),
                'packet_id': row['packet_id'],
                'source': row['source'],
                'target': row['target'],
                'content': row['content'],
                'role_combination': role_combination,
                'is_source': is_source,
                'is_target': is_target,
                'is_mentioned': is_mentioned
            })
    if not nadia_communications:
        fig = go.Figure()
        fig.add_annotation(
            text="No communications involving Nadia Conti found",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Nadia Conti Communication Timeline",
            xaxis_title="Date",
            yaxis_title="Packet ID",
            height=600,
            width=900
        )
        return fig
    nadia_df = pd.DataFrame(nadia_communications)
    nadia_df = nadia_df.sort_values(['timestamp', 'packet_id'])
    role_colors = {
        'Source + Target': '#d62728',
        'Source': '#1f77b4',
        'Target': '#ff7f0e',
        'Mentioned': '#2ca02c',
        'Source + Target + Mentioned': '#e377c2',
        'Target + Mentioned': '#8c5644'
    }
    fig = go.Figure()
    # Show each point at (timestamp, packet_id)
    for idx, row in nadia_df.iterrows():
        color = role_colors.get(row['role_combination'], '#808080')
        hover_text = f"<b>Date:</b> {row['date'].strftime('%Y-%m-%d')}<br>" \
                     f"<b>Time:</b> {row['timestamp'].strftime('%H:%M')}<br>" \
                     f"<b>Packet ID:</b> {row['packet_id']}<br>" \
                     f"<b>Source:</b> {row['source']}<br>" \
                     f"<b>Target:</b> {row['target']}<br>" \
                     f"<b>Role:</b> {row['role_combination']}<br>" \
                     f"<b>Content:</b> {row['content'][:100]}..."
        fig.add_trace(go.Scatter(
            x=[row['timestamp']],
            y=[f"Packet {row['packet_id']}"],
            mode='markers',
            marker=dict(
                size=12,
                color=color,
                symbol='circle'
            ),
            name=row['role_combination'],
            text=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    # Update y-axis to show all packet IDs
    unique_packet_ids = sorted(nadia_df['packet_id'].unique())
    fig.update_layout(
        title="Nadia Conti Communication Timeline - Colored by Role Combinations",
        xaxis_title="Date",
        yaxis_title="Packet ID",
        height=1000,
        width=1400,
        hovermode='closest',
        yaxis=dict(
            tickvals=[f"Packet {pid}" for pid in unique_packet_ids],
            ticktext=[f"Packet {pid}" for pid in unique_packet_ids],
            showgrid=True,
            gridcolor='lightblue',
            gridwidth=1,
            zeroline=False,
            showticklabels=True
        ),
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            tickangle=45,
            showgrid=True,
            gridcolor='lightblue',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12)
        ),
        legend=dict(
            title='Role Combinations',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=100)
    )
    # Add legend manually with counts
    role_counts = nadia_df['role_combination'].value_counts()
    for role_combo, color in role_colors.items():
        if role_combo in nadia_df['role_combination'].values:
            count = role_counts.get(role_combo, 0)
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"{role_combo} ({count})",
                showlegend=True
            ))
    return fig

# =====================
# Nadia Conti LLM Packet Summary Callback
# =====================
from dash.dependencies import Input, Output, State
@app.callback(
    [Output("llm-summary-container", "children"),
     Output("llm-summary-checkbox", "style")],
    [Input("nadia-gantt-chart", "clickData"),
     Input("llm-summary-checkbox", "value")]
)
def update_nadia_packet_summary(clickData, llm_checkbox):
    # Always show the container now
    style = {"display": "block", "marginTop": "20px", "padding": "10px", "borderRadius": "5px"}

    # If no packet is selected, show a message and disable LLM summary
    if not clickData or "points" not in clickData or not clickData["points"]:
        return [html.Div([html.P("Click a packet in the timeline above to view its network analysis and generate an LLM summary.")]), style]

    # Extract packet_id from clickData
    point = clickData["points"][0]
    packet_id = point.get("y")
    try:
        packet_id = int(str(packet_id).replace("Packet ", ""))
    except Exception:
        return [html.Div([html.P("Could not determine packet ID from click.")]), style]

    # Get all messages for this packet
    packet_df = comms_df[comms_df['packet_id'] == packet_id]
    if packet_df.empty:
        return [html.Div([html.P(f"No messages found for packet {packet_id}.")]), style]

    # Basic info
    basic_info = html.Div([
        html.H4("Packet Summary"),
        html.P(f"Packet ID: {packet_id}"),
        html.P(f"Messages in packet: {len(packet_df)}"),
    ])

    # Network analysis (simple undirected graph)
    import networkx as nx
    import plotly.graph_objects as go
    G = nx.Graph()
    for _, row in packet_df.iterrows():
        G.add_edge(row['source'], row['target'])
    betweenness = nx.betweenness_centrality(G) if len(G) > 0 else {}
    pos = nx.spring_layout(G, seed=42, k=1.5) if len(G) > 1 else {n: (0, 0) for n in G.nodes()}
    edge_traces = []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    node_x, node_y, node_text, node_size, node_color, hover_texts = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        centrality = betweenness.get(node, 0)
        node_size.append(20 + 80 * centrality)
        node_color.append(f'rgba(255, {int(255 * (1 - centrality))}, {int(255 * (1 - centrality))}, 0.8)')
        hover_texts.append(f"{node}<br>Betweenness Centrality: {centrality:.3f}")
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False
    )
    subgraph_fig = go.Figure(data=edge_traces + [node_trace],
                            layout=go.Layout(
                                title=f"Network Analysis for Packet {packet_id} - Betweenness Centrality",
                                showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=10),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=400, width=600
                            ))
    centrality_info = html.Div([
        html.H5("Betweenness Centrality Analysis:"),
        html.Ul([
            html.Li(f"{node}: {centrality:.3f}") 
            for node, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        ])
    ])
    network_analysis = html.Div([
        html.Hr(),
        html.H4("Network Analysis"),
        dcc.Graph(figure=subgraph_fig, config={'displayModeBar': False}),
        centrality_info
    ])

    summary_content = [basic_info, network_analysis]

    # If LLM summary is requested
    if llm_checkbox and "generate" in llm_checkbox:
        prompt = f"""
You are a skilled agent, good at summarizing and solving a case where a suspected entity is using loopholes to seek expedited approval to violate certain environmental channels.
From the incoming list of messages {packet_df['content'].tolist()} and [{packet_id}],
Your job is to summarize all content in the packet as succinctly as possible, and identify the role Nadia conti has played.
Return a dictionary with the key 'packet_id' followed by two lists, which contain ['summary of events in the packet','role of nadia conti in the packet']

Return only the dictionary. Do not return anything else.
        """
        try:
            llm_response = get_chatbot_response(prompt, chatbot_graph, chatbot_config)
            import ast
            parsed = ast.literal_eval(llm_response)
            summary = parsed.get('summary of events in the packet', 'No summary available')
            nadia_role = parsed.get('role of nadia conti in the packet', 'No role information available')
            llm_analysis = html.Div([
                html.Hr(),
                html.H4(f"LLM Analysis"),
                html.Pre(f"Summary of events:\n{summary}\n\nRole of Nadia Conti:\n{nadia_role}", style={"whiteSpace": "pre-wrap", "fontSize": "14px"})
            ])
        except Exception as e:
            llm_analysis = html.Div([
                html.Hr(),
                html.H4(f"LLM Analysis"),
                html.P(f"Error generating LLM summary: {e}")
            ])
        summary_content.append(llm_analysis)

    return [summary_content, style]

# =====================
# Pseudonyms Sankey Helper
# =====================
def build_pseudonyms_sankey(min_threshold=0.6, max_threshold=0.7):
    """
    Build Sankey chart for pseudonyms showing flow between 4 columns:
    Source_ID_source → Source_ID_target → Target_ID_source → Target_ID_target
    Each Source_ID_source gets its own line, and each Source_ID_target pair gets separate lines
    """
    # Filter by similarity range
    filtered_df = pseduonyms_df[(pseduonyms_df['cosine_similarity'] >= min_threshold) & 
                               (pseduonyms_df['cosine_similarity'] <= max_threshold)].copy()
    
    if filtered_df.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data found in similarity range {min_threshold:.2f} - {max_threshold:.2f}<br>Available range: {pseduonyms_df['cosine_similarity'].min():.2f} - {pseduonyms_df['cosine_similarity'].max():.2f}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title_text=f"Pseudonyms Flow Analysis (Similarity Range: {min_threshold:.2f} - {max_threshold:.2f})",
            font_size=12,
            height=800,
            width=1200
        )
        return fig
    
    # Group by Source_ID_source and sort for consistent ordering
    source_groups = filtered_df.groupby('Source_ID_source').size().sort_values(ascending=False)
    
    # Create unique labels with separate lines for each source-target pair
    labels = []
    label_to_index = {}
    idx = 0
    
    # Column 1: Source_ID_source (each source gets its own line)
    for source_entity in source_groups.index:
        if source_entity not in label_to_index:
            label_to_index[source_entity] = idx
            labels.append(f"Source: {source_entity}")
            idx += 1
    
    # Column 2: Source_ID_target (each unique source-target pair gets its own line)
    for source_entity in source_groups.index:
        # Get all unique source-target pairs for this source
        source_target_pairs = filtered_df[filtered_df['Source_ID_source'] == source_entity][['Source_ID_source', 'Source_ID_target']].drop_duplicates()
        for _, pair in source_target_pairs.iterrows():
            target_entity = pair['Source_ID_target']
            # Create unique identifier for this source-target pair
            pair_id = f"{source_entity}_{target_entity}"
            if pair_id not in label_to_index:
                label_to_index[pair_id] = idx
                labels.append(f"Source Target: {target_entity}")
                idx += 1
    
    # Column 3: Target_ID_source (each unique target-source pair gets its own line)
    for source_entity in source_groups.index:
        # Get all unique target-source pairs for this source
        target_source_pairs = filtered_df[filtered_df['Source_ID_source'] == source_entity][['Source_ID_target', 'Target_ID_source']].drop_duplicates()
        for _, pair in target_source_pairs.iterrows():
            target_source_entity = pair['Target_ID_source']
            # Create unique identifier for this target-source pair
            pair_id = f"{pair['Source_ID_target']}_{target_source_entity}"
            if pair_id not in label_to_index:
                label_to_index[pair_id] = idx
                labels.append(f"Target Source: {target_source_entity}")
                idx += 1
    
    # Column 4: Target_ID_target (each unique target-target pair gets its own line)
    for source_entity in source_groups.index:
        # Get all unique target-target pairs for this source
        target_target_pairs = filtered_df[filtered_df['Source_ID_source'] == source_entity][['Target_ID_source', 'Target_ID_target']].drop_duplicates()
        for _, pair in target_target_pairs.iterrows():
            target_target_entity = pair['Target_ID_target']
            # Create unique identifier for this target-target pair
            pair_id = f"{pair['Target_ID_source']}_{target_target_entity}"
            if pair_id not in label_to_index:
                label_to_index[pair_id] = idx
                labels.append(f"Target: {target_target_entity}")
                idx += 1
    
    # Create sources, targets, and values for Sankey
    sources = []
    targets = []
    values = []
    hover_texts = []
    
    for _, row in filtered_df.iterrows():
        # Flow 1: Source_ID_source → Source_ID_target (using pair identifier)
        source_idx = label_to_index[row['Source_ID_source']]
        pair_id = f"{row['Source_ID_source']}_{row['Source_ID_target']}"
        target_idx = label_to_index[pair_id]
        similarity = row['cosine_similarity']
        
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(similarity)
        hover_texts.append(f"Similarity: {similarity:.3f}<br>Source Message: {row['source_message'][:100]}...")
        
        # Flow 2: Source_ID_target → Target_ID_source (using pair identifier)
        source_pair_id = f"{row['Source_ID_source']}_{row['Source_ID_target']}"
        target_pair_id = f"{row['Source_ID_target']}_{row['Target_ID_source']}"
        source_idx = label_to_index[source_pair_id]
        target_idx = label_to_index[target_pair_id]
        
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(similarity)
        hover_texts.append(f"Similarity: {similarity:.3f}<br>Source Message: {row['source_message'][:100]}...<br>Target Message: {row['target_message'][:100]}...")
        
        # Flow 3: Target_ID_source → Target_ID_target (using pair identifier)
        source_pair_id = f"{row['Source_ID_target']}_{row['Target_ID_source']}"
        target_pair_id = f"{row['Target_ID_source']}_{row['Target_ID_target']}"
        source_idx = label_to_index[source_pair_id]
        target_idx = label_to_index[target_pair_id]
        
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(similarity)
        hover_texts.append(f"Similarity: {similarity:.3f}<br>Target Message: {row['target_message'][:100]}...")
    
    # Create Sankey figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        )
    )])
    
    fig.update_layout(
        title_text=f"Pseudonyms Flow Analysis (Similarity Range: {min_threshold:.2f} - {max_threshold:.2f}) - Separate Lines per Source-Target Pair",
        font_size=12,
        height=800,
        width=1200
    )
    
    return fig

def create_pseudonyms_tab_layout():
    """Create the layout for the pseudonyms tab"""
    return html.Div([
        html.H3("Pseudonyms Sankey Analysis", style={"textAlign": "center", "marginTop": "32px", "marginBottom": "16px"}),
        html.P("This Sankey chart shows the flow of communication patterns between entities. The 4 columns represent: Source → Source Target → Target Source → Target. Hover over links to see message details.", 
               style={"textAlign": "center", "marginBottom": "24px", "fontSize": "1.1em"}),
        
        # Controls
        html.Div([
            html.Label("Similarity Range:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.RangeSlider(
                id="similarity-threshold-slider",
                min=0.6,
                max=0.95,
                step=0.01,
                value=[0.79, 0.8],
                marks={0.6: "0.6", 0.65: "0.65", 0.7: "0.7", 0.75: "0.75", 0.8: "0.8", 0.85: "0.85", 0.9: "0.9", 0.95: "0.95"},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False
            )
        ], style={"marginBottom": "24px", "width": "100%"}),
        
        # Sankey chart
        dcc.Graph(
            id='pseudonyms-sankey',
            style={'width': '100%', 'height': '800px'}
        ),
        
        # Statistics
        html.Div(id="pseudonyms-stats", style={"marginTop": "24px", "textAlign": "center"}),
        
        # Details table
        html.Div([
            html.H4("High Similarity Message Pairs", style={"marginTop": "32px", "marginBottom": "16px"}),
            dash_table.DataTable(
                id="pseudonyms-details-table",
                columns=[
                    {"name": "Similarity", "id": "cosine_similarity"},
                    {"name": "Source Pair", "id": "source_pair"},
                    {"name": "Target Pair", "id": "target_pair"},
                    {"name": "Source Message", "id": "source_message"},
                    {"name": "Target Message", "id": "target_message"}
                ],
                style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'fontSize': '0.9em', 'minWidth': '100px', 'maxWidth': '300px', 'whiteSpace': 'normal'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f7f7fa'},
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        ], style={"marginTop": "32px"})
    ], className="p-4")

# Replace the Temporal Patterns tab layout
app.layout = dbc.Container([
    html.H1("DOCK-G: Decoding Oceanus's Corruption Knowledge-Graphs", className="my-4 text-center"),
    chatbot_card,
    dcc.Tabs([
        dcc.Tab(label="Temporal Patterns", children=[motif_tab_layout()]),
        dcc.Tab(label="Vessels & People", children=[
            html.Div([
                html.H4("Entity Communication Network"),
                html.P("This interactive network shows which entities communicate most frequently. Groups that talk repeatedly amongst themselves are visually prominent."),
                html.Iframe(
                    id="entity-network-iframe",
                    srcDoc=get_network_html(build_entity_network(comms_df, topic_model_df), {}),
                    width="100%",
                    height="700px",
                    style={"border": "none"}
                ),
                html.H3("Communication Networks by Topic (Interactive)", className="my-4 text-center"),
                html.P("Zoom to see subgraphs and entity connections more clearly", style={"textAlign": "center", "color": "#555", "fontSize": "1.1em", "marginBottom": "8px"}),
                html.Label("Centrality Measure:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="centrality-dropdown",
                    options=centrality_options,
                    value="betweenness",
                    clearable=False,
                    style={"width": "300px", "marginBottom": "24px"}
                ),
                html.Div(id="topic-pyvis-grid"),
                html.Hr(),
                html.H3("Entity Subgraph Explorer", className="my-4 text-center"),
                html.P("Zoom to see subgraphs and entity connections more clearly", style={"textAlign": "center", "color": "#555", "fontSize": "1.1em", "marginBottom": "8px"}),
                html.Label("Select Entities:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="entity-dropdown",
                    options=entity_options,
                    multi=True,
                    value=["boss", "davis"],
                    style={"width": "600px", "marginBottom": "24px"}
                ),
                html.Div(id="entity-subgraph-section")
            ], className="p-4")
        ]),
        dcc.Tab(label="Pseudonyms", children=[create_pseudonyms_tab_layout()]),
        dcc.Tab(label="Who is Nadia Conti", children=[create_nadia_conti_tab_layout()]),
    ],
    parent_className="mb-3",
    className="mb-3",
    ),
    # Add hidden llm-summary-checkbox so Dash always registers it
    html.Div([
        dcc.Checklist(
            id="llm-summary-checkbox",
            options=[{"label": "Generate LLM Summary", "value": "generate"}],
            value=[],
            style={"display": "none"}
        )
    ], style={"display": "none"}),
    # Add hidden nadia-packet-summary so Dash always registers it
    # html.Div(id="nadia-packet-summary", style={"display": "none"})
], fluid=True)

# Collapsible explainer callback
def register_callbacks(app):
    @app.callback(
        Output("collapse-explainer", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse-explainer", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

# Dash callback to update grid
@app.callback(
    Output("topic-pyvis-grid", "children"),
    [Input("centrality-dropdown", "value")]
)
def update_topic_pyvis_grid(centrality_measure):
    return topic_pyvis_grid(centrality_measure)

# Dash callback for entity subgraph explorer
@app.callback(
    Output("entity-subgraph-section", "children"),
    [Input("centrality-dropdown", "value"), Input("entity-dropdown", "value")]
)
def update_entity_subgraph_section(centrality_measure, selected_entities):
    return entity_subgraph_section(centrality_measure, selected_entities or [])

# Chatbot callback: manage chat history and render chat bubbles
@app.callback(
    [Output("chatbot-chat-area", "children"), Output("chatbot-history", "data"), Output("chatbot-input", "value"), Output("chatbot-loading", "style"), Output("chatbot-loading-interval", "disabled")],
    [Input("chatbot-send", "n_clicks"), Input("chatbot-loading-interval", "n_intervals")],
    [State("chatbot-input", "value"), State("chatbot-history", "data")],
    prevent_initial_call=True
)
def update_chatbot_chat_area(n_clicks, n_intervals, value, history):
    global chatbot_session_count, chatbot_config
    
    # Get the trigger that caused this callback
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Default loading style (hidden)
    loading_style = {
        "display": "none",
        "justifyContent": "center",
        "alignItems": "center",
        "padding": "12px",
        "background": "#f8f9fa",
        "borderRadius": "8px",
        "marginTop": "8px",
        "border": "1px solid #e9ecef"
    }
    
    if not history:
        history = []
    
    # Handle send button click
    if trigger_id == "chatbot-send" and n_clicks and value:
        # Show loading spinner
        loading_style["display"] = "flex"
        
        # Increment session count
        chatbot_session_count += 1
        
        # Check if session needs to be reset
        if chatbot_session_count >= MAX_MESSAGES_PER_SESSION:
            # Reset session with new config
            chatbot_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            chatbot_session_count = 0
            
            # Add session reset message
            history.append({"role": "bot", "text": "🔄 **Session Reset**: Your conversation session has been reset after 5 messages. Starting fresh with a new context!"})
        
        # Append user message
        history.append({"role": "user", "text": value})
        
        # Get chatbot response using the LangGraph logic
        try:
            bot_response = get_chatbot_response(value, chatbot_graph, chatbot_config)
        except Exception as e:
            bot_response = f"Error processing your question: {str(e)}"
        
        history.append({"role": "bot", "text": bot_response})
        
        # Enable interval
        interval_disabled = False
    
    # Handle interval trigger (hide loading)
    elif trigger_id == "chatbot-loading-interval" and n_intervals > 0:
        # Hide loading spinner
        loading_style["display"] = "none"
        interval_disabled = True
    
    # Render chat bubbles
    chat_bubbles = []
    for msg in history:
        align = "flex-end" if msg["role"] == "user" else "flex-start"
        bubble_color = "#0074D9" if msg["role"] == "user" else "#e9ecef"
        text_color = "#fff" if msg["role"] == "user" else "#222"
        if msg["role"] == "user":
            border_radius = "18px"
        else:
            border_radius = "18px 18px 18px 4px"
        bubble_style = {
            "maxWidth": "70%",
            "alignSelf": align,
            "background": bubble_color,
            "color": text_color,
            "padding": "12px 18px",
            "borderRadius": border_radius,
            "marginBottom": "2px",
            "fontSize": "1.08em",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
            "wordBreak": "break-word"
        }
        chat_bubbles.append(html.Div(msg["text"], style=bubble_style))
    
    return chat_bubbles, history, "", loading_style, interval_disabled

# Callback to update session counter
@app.callback(
    Output("chatbot-session-counter", "children"),
    [Input("chatbot-send", "n_clicks")],
    prevent_initial_call=True
)
def update_session_counter(n_clicks):
    global chatbot_session_count
    if n_clicks:
        return f"{chatbot_session_count}/{MAX_MESSAGES_PER_SESSION}"
    return "0/5"



# =====================
# Callback for Motif Length Slider and Days Dropdown
# =====================

def build_motif_pyvis_network(motifs, counts):
    """
    motifs: dict of motif tuple -> set of days
    counts: dict of motif tuple -> count
    Returns: HTML string for PyVis network
    """
    import networkx as nx
    # Use all motifs, not just top N
    sorted_motifs = list(counts.keys())
    G = nx.DiGraph()
    for motif in sorted_motifs:
        for i in range(len(motif) - 1):
            src, tgt = motif[i], motif[i+1]
            G.add_edge(src, tgt)
    # Build PyVis network
    net = Network(height="700px", width="100%", notebook=False, directed=True)
    net.barnes_hut(gravity=-30000, central_gravity=0.1, spring_length=300, spring_strength=0.005, damping=0.25, overlap=0.8)
    for node in G.nodes():
        net.add_node(node, label=node, color="#4F8EF7", size=30, font={"size": 24, "color": "#111", "face": "Arial Black", "bold": True})
    for src, tgt in G.edges():
        net.add_edge(src, tgt, width=2, color="#FF4136", arrows="to")
    return net.generate_html()

# PyVis motif network callback
@app.callback(
    Output("motif-network-iframe", "srcDoc"),
    [Input("motif-length-slider", "value"), Input("motif-days-dropdown", "value")]
)
def update_motif_network_pyvis(min_length, min_days):
    filtered_motifs = {m: days for m, days in repeating_motifs_all.items() if len(m) == min_length and len(days) >= min_days}
    filtered_counts = {m: motif_counts_all[m] for m in filtered_motifs}
    if not filtered_motifs:
        return "<html><body><h3>No motifs of this length and day count found.</h3></body></html>"
    return build_motif_pyvis_network(filtered_motifs, filtered_counts)

# Callback to update motif details table
@app.callback(
    Output("motif-details-table-container", "children"),
    [Input("motif-length-slider", "value"), Input("motif-days-dropdown", "value")]
)
def update_motif_details_table(min_length, min_days):
    filtered_motifs = {m: days for m, days in repeating_motifs_all.items() if len(m) == min_length and len(days) >= min_days}
    return build_motif_details_table(filtered_motifs, min_length, min_days)

# Update sunburst chart callback
@app.callback(
    Output("sunburst-temporal-patterns", "figure"),
    [Input("sunburst-entity-dropdown", "value")]
)
def update_sunburst_figure(selected_entities_sunburst):
    return build_sunburst_figure(selected_entities_sunburst)

# Callback to update windrose chart based on selected entity
@app.callback(
    Output("mako-windrose-graph", "figure"),
    [Input("windrose-entity-dropdown", "value")]
)
def update_mako_windrose_figure(selected_entity):
    return build_mako_windrose_figure(selected_entity)

# Register windrose callback
# Remove the callback for mako-windrose-graph.figure since there are no Inputs

# Pseudonyms callbacks
@app.callback(
    [Output("pseudonyms-sankey", "figure"),
     Output("pseudonyms-stats", "children"),
     Output("pseudonyms-details-table", "data")],
    [Input("similarity-threshold-slider", "value")]
)
def update_pseudonyms_sankey(threshold_range):
    min_threshold, max_threshold = threshold_range
    fig = build_pseudonyms_sankey(min_threshold, max_threshold)
    
    # Calculate statistics
    filtered_df = pseduonyms_df[(pseduonyms_df['cosine_similarity'] >= min_threshold) & 
                               (pseduonyms_df['cosine_similarity'] <= max_threshold)]
    total_pairs = len(filtered_df)
    avg_similarity = filtered_df['cosine_similarity'].mean() if not filtered_df.empty else 0
    
    stats = html.Div([
        html.H5(f"Sankey Statistics (Range: {min_threshold:.2f} - {max_threshold:.2f})"),
        html.P(f"Total pairs in range: {total_pairs}"),
        html.P(f"Average similarity: {avg_similarity:.3f}"),
        html.P(f"Unique entities: {len(set(filtered_df['Source_ID_source'].unique()) | set(filtered_df['Source_ID_target'].unique()) | set(filtered_df['Target_ID_source'].unique()) | set(filtered_df['Target_ID_target'].unique()))}")
    ])
    
    # Prepare table data
    table_data = []
    for _, row in filtered_df.iterrows():
        table_data.append({
            'cosine_similarity': f"{row['cosine_similarity']:.3f}",
            'source_pair': f"{row['Source_ID_source']} → {row['Source_ID_target']}",
            'target_pair': f"{row['Target_ID_source']} → {row['Target_ID_target']}",
            'source_message': row['source_message'][:150] + '...' if len(row['source_message']) > 150 else row['source_message'],
            'target_message': row['target_message'][:150] + '...' if len(row['target_message']) > 150 else row['target_message']
        })
    
    return fig, stats, table_data

register_callbacks(app)

# =====================
# Nadia Conti Callbacks
# =====================

@app.callback(
    Output("nadia-gantt-chart", "figure"),
    [Input("nadia-gantt-chart", "clickData")]
)
def update_nadia_gantt_figure(click_data):
    return build_nadia_gantt_figure()

# Store last clicked packet from Gantt chart
@app.callback(
    Output("nadia-last-clicked-packet", "data"),
    [Input("nadia-gantt-chart", "clickData")]
)
def store_last_clicked_packet(clickData):
    if clickData and "points" in clickData and clickData["points"]:
        point = clickData["points"][0]
        packet_id = point.get("y")
        try:
            packet_id = int(str(packet_id).replace("Packet ", ""))
            return packet_id
        except Exception:
            return None
    return None

# ========== DEBUG: Print motifs involving mrs. money, boss, or the middleman ==========
if __name__ == "__main__":
    # Existing app.run(debug=True) call
    app.run(debug=True)



