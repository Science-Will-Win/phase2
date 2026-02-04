"""
Biomni dummy tools for biomedical research.
These tools use LLM to generate dynamic results based on input.
"""

import random
from typing import Dict, Any, List
from tools.base import Tool, ToolParameter, register_tool


def generate_with_llm(prompt: str, max_tokens: int = 150, system_prompt: str = None) -> str:
    """Use global model to generate specific data for tool results.
    
    Args:
        prompt: The prompt for LLM generation
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt for context
        
    Returns:
        Generated text string, or empty string on failure
    """
    try:
        # Import global model from inference (lazy import to avoid circular dependency)
        from inference import global_model, global_tokenizer, global_args
        
        if global_model is None or global_tokenizer is None:
            return ""
        
        # Build messages with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        text = global_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = global_tokenizer(text, return_tensors="pt")
        if hasattr(global_model, 'device'):
            inputs = {k: v.to(global_model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = global_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=global_tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = global_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    except Exception as e:
        print(f"[DEBUG] LLM generation failed in tool: {e}")
        return ""


# Fallback gene lists for when LLM is not available (32 genes per category)
FALLBACK_GENES = {
    "t cell": [
        "PDCD1", "HAVCR2", "TOX", "LAG3", "TIGIT", "CTLA4", "CD8A", "IFNG",
        "CD28", "ICOS", "BTLA", "CD160", "CD244", "ENTPD1", "NT5E", "EOMES",
        "TBX21", "PRDM1", "IRF4", "BATF", "NR4A1", "NR4A2", "NR4A3", "TCF7",
        "LEF1", "ID2", "ID3", "NFATC1", "NFATC2", "TOX2", "CXCL13", "IL10"
    ],
    "cancer": [
        "TP53", "KRAS", "EGFR", "BRAF", "PIK3CA", "MYC", "RB1", "PTEN",
        "APC", "BRCA1", "BRCA2", "CDK4", "CDK6", "CDKN2A", "CDKN2B", "MDM2",
        "ERBB2", "FGFR1", "FGFR2", "FGFR3", "MET", "ALK", "ROS1", "RET",
        "NRAS", "HRAS", "NF1", "NF2", "VHL", "STK11", "SMAD4", "ARID1A"
    ],
    "immune": [
        "IL2", "IFNG", "TNF", "CD4", "CD8A", "FOXP3", "GZMB", "PRF1",
        "IL6", "IL10", "IL17A", "IL21", "IL23A", "TGFB1", "CXCL8", "CCL2",
        "CCR7", "CXCR3", "CXCR5", "CCR5", "CD80", "CD86", "CD40", "CD40LG",
        "TNFRSF9", "TNFRSF4", "TNFRSF18", "CD27", "CD70", "ICOS", "ICOSLG", "B2M"
    ],
    "default": [
        "PDCD1", "HAVCR2", "TOX", "LAG3", "TIGIT", "CTLA4", "CD28", "ICOS",
        "CD8A", "CD4", "IFNG", "TNF", "IL2", "GZMB", "PRF1", "FOXP3",
        "EOMES", "TBX21", "TCF7", "LEF1", "BATF", "IRF4", "PRDM1", "NR4A1",
        "ENTPD1", "NT5E", "CD160", "CD244", "BTLA", "CXCL13", "IL10", "TOX2"
    ]
}


def get_fallback_genes(query: str) -> List[str]:
    """Get fallback gene list based on query keywords."""
    query_lower = query.lower()
    for keyword, genes in FALLBACK_GENES.items():
        if keyword in query_lower:
            return genes
    return FALLBACK_GENES["default"]


# Dummy results based on HTML demo (returns results in example order)
DUMMY_RESULTS = {
    "pubmed_search": {
        "title": "Found 847 related papers",
        "details": [
            "Molecular regulators of T cell exhaustion (Nature, 2024)",
            "Key markers: PD-1, TIM-3, LAG-3, TIGIT",
            "Extracted 32 candidate regulatory genes"
        ],
        "tokens": 2450,
        "duration": "3.2s"
    },
    "ncbi_gene": {
        "title": "Retrieved info for 32 genes",
        "details": [
            "PDCD1 (PD-1): inhibitory receptor, chr 2q37.3",
            "HAVCR2 (TIM-3): immunoglobulin domain, chr 5q33.3",
            "TOX: key exhaustion transcription factor"
        ],
        "tokens": 1820,
        "duration": "2.1s"
    },
    "crispr_designer": {
        "title": "Designed 96 sgRNAs",
        "details": [
            "Average efficiency score: 0.72",
            "Off-targets: all below 3",
            "GC content: 40-60%"
        ],
        "tokens": 3240,
        "duration": "4.8s",
        "has_graph": True,
        "graph_type": "efficiency"
    },
    "protocol_builder": {
        "title": "Generated 6-week experimental protocol",
        "details": [
            "Week 1-2: Library cloning and lentivirus production",
            "Week 3-4: T cell transduction and selection",
            "Week 5-6: Stimulation analysis and sequencing"
        ],
        "tokens": 2680,
        "duration": "2.9s",
        "has_graph": True,
        "graph_type": "timeline"
    }
}


@register_tool
class PubmedSearchTool(Tool):
    """Tool for searching PubMed literature database."""
    
    name = "pubmed_search"
    description = "Searches the PubMed literature database for relevant papers. Use to find latest literature on research topics, genes, diseases, etc."
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="Query string to search (e.g., 'T cell exhaustion CRISPR screen')",
            required=True
        ),
        ToolParameter(
            name="max_results",
            type="number",
            description="Maximum number of results to return",
            required=False,
            default=100
        )
    ]
    
    def execute(self, query: str, max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute PubMed search with LLM-generated gene list."""
        # Try to generate relevant gene names using LLM (32 genes)
        gene_prompt = f"""Based on the biomedical search query "{query}", list 32 relevant gene symbols that would be found in related research papers.
Include key regulators, transcription factors, signaling molecules, and effector genes.
Output ONLY the gene symbols separated by commas, nothing else.
Example: PDCD1, HAVCR2, TOX, LAG3, TIGIT, CTLA4, CD8A, IFNG, ..."""
        
        genes_text = generate_with_llm(gene_prompt, max_tokens=300)
        
        # Parse gene list from LLM output
        gene_list = []
        if genes_text:
            # Clean up and split
            for g in genes_text.replace('\n', ',').split(','):
                gene = g.strip().upper()
                # Filter out non-gene-like strings
                if gene and len(gene) <= 10 and gene.isalnum():
                    gene_list.append(gene)
        
        # Use fallback if LLM failed or returned too few genes
        if len(gene_list) < 10:
            gene_list = get_fallback_genes(query)
        
        # Ensure exactly 32 genes (pad with fallback if needed)
        if len(gene_list) < 32:
            fallback = get_fallback_genes(query)
            for g in fallback:
                if g not in gene_list and len(gene_list) < 32:
                    gene_list.append(g)
        gene_list = gene_list[:32]
        
        # Generate detailed summary using LLM
        summary_prompt = f"""Based on a PubMed search for "{query}", provide a brief research summary:
1. Key research trends (1-2 sentences)
2. Core findings (1-2 sentences)
3. Clinical significance (1 sentence)
Output in English, concise and scientific."""
        
        llm_summary = generate_with_llm(summary_prompt, max_tokens=200)
        
        # Build result with dynamic gene list and LLM summary
        num_papers = random.randint(500, 1200)
        
        # Parse LLM summary into details
        details = [f"Search query: {query[:50]}"]
        if llm_summary:
            # Split by newlines or numbered items
            for line in llm_summary.replace('\n', '|').split('|'):
                line = line.strip()
                if line and len(line) > 5:
                    details.append(line[:150])
        
        # Add gene information
        details.extend([
            f"Key candidate genes: {', '.join(gene_list[:8])}...",
            f"Extracted {len(gene_list)} candidate regulatory genes"
        ])
        
        result = {
            "title": f"Found {num_papers} related papers",
            "details": details,
            "candidate_genes": gene_list,  # Structured data for next step
            "query": query,
            "max_results": max_results,
            "tokens": random.randint(2000, 3000),
            "duration": f"{random.uniform(2.0, 4.0):.1f}s"
        }
        
        return {
            "success": True,
            "tool": self.name,
            "thought": f"Searching latest literature for '{query}' and extracting {len(gene_list)} candidate genes",
            "action": f'PubMed API call: query="{query}", analyzing {num_papers} results',
            "result": result
        }


@register_tool
class NcbiGeneTool(Tool):
    """Tool for querying NCBI Gene database."""
    
    name = "ncbi_gene"
    description = "Queries the NCBI Gene database for gene information. Use to check gene function, location, and related information."
    parameters = [
        ToolParameter(
            name="genes",
            type="array",
            description="List of genes to query (e.g., ['PDCD1', 'HAVCR2', 'TOX'])",
            required=True
        )
    ]
    
    def execute(self, genes: List[str], **kwargs) -> Dict[str, Any]:
        """Execute NCBI Gene query with LLM-generated descriptions."""
        details = []
        summary_info = []
        
        if genes:
            # Generate detailed descriptions for all 32 genes
            key_genes = genes[:32]
            gene_str = ', '.join(key_genes)
            desc_prompt = f"""For each gene: {gene_str}
Provide detailed information:
- Gene symbol: function, chromosome location, related signaling pathways, disease associations
Format: GENE: description (one line each)
Be specific and scientific."""
            
            max_tokens = max(1200, len(key_genes) * 40)
            desc_text = generate_with_llm(desc_prompt, max_tokens=max_tokens)
            
            # Parse descriptions
            if desc_text and ':' in desc_text:
                for part in desc_text.replace('\n', '|').split('|'):
                    part = part.strip()
                    if part and ':' in part and len(part) > 5:
                        details.append(part[:150])
            
            # Fallback if parsing failed - 32 descriptions for all genes
            if len(details) < len(key_genes):
                fallback_descriptions = [
                    ("immune checkpoint inhibitory receptor", "chr 2q37.3", "key T cell exhaustion marker"),
                    ("inhibitory receptor TIM-3", "chr 5q33.3", "immune regulation"),
                    ("key exhaustion transcription factor", "chr 8q12.1", "T cell differentiation regulation"),
                    ("inhibitory receptor", "chr 12p13.32", "immune checkpoint"),
                    ("co-inhibitory receptor", "chr 3q13.31", "NK/T cell function"),
                    ("co-stimulatory molecule", "chr 2q33.2", "T cell activation inhibition"),
                    ("cytotoxic T cell marker", "chr 2p11.2", "CD8+ T cells"),
                    ("cytokine", "chr 12q15", "Th1 immune response"),
                    ("co-stimulatory receptor", "chr 2q33.2", "T cell activation"),
                    ("inhibitory receptor", "chr 21q22.3", "NK cell regulation"),
                    ("transcription factor", "chr 10q21.2", "Th17 differentiation"),
                    ("transcription factor", "chr 17q21.32", "Treg development"),
                    ("nuclear receptor", "chr 12q13.3", "T cell function regulation"),
                    ("nuclear receptor", "chr 2q24.1", "immune homeostasis"),
                    ("nuclear receptor", "chr 9q22.31", "cell survival"),
                    ("transcription factor", "chr 10q25.2", "memory T cells"),
                    ("transcription factor", "chr 4q27", "T cell differentiation"),
                    ("transcription factor", "chr 2p25.1", "effector T cells"),
                    ("transcription factor", "chr 1p36.12", "innate immunity"),
                    ("transcription factor", "chr 6p21.1", "T cell activation"),
                    ("signal transduction", "chr 18q21.33", "immunosuppression"),
                    ("signal transduction", "chr 18q21.33", "TGF-beta pathway"),
                    ("chemokine receptor", "chr 17q21.2", "lymphocyte migration"),
                    ("chemokine receptor", "chr Xq13.1", "Th1 migration"),
                    ("chemokine receptor", "chr 11q23.3", "B cell migration"),
                    ("chemokine receptor", "chr 3p21.31", "T cell migration"),
                    ("co-stimulatory molecule", "chr 3q13.33", "APC activation"),
                    ("co-stimulatory molecule", "chr 3q13.33", "T cell priming"),
                    ("co-stimulatory ligand", "chr 20q13.12", "T cell activation"),
                    ("TNF receptor", "chr 1p36.23", "T cell co-stimulation"),
                    ("TNF receptor", "chr 1p36.33", "T cell survival"),
                    ("TNF receptor", "chr 1p36.33", "Treg function"),
                ]
                details = []
                for i, gene in enumerate(key_genes):
                    desc = fallback_descriptions[i % len(fallback_descriptions)]
                    details.append(f"{gene}: {desc[0]}, {desc[1]}, {desc[2]}")
            
            # Generate summary analysis
            summary_prompt = f"""Given {len(genes)} genes related to immune regulation, provide a brief analysis:
1. Main functional categories (checkpoints, transcription factors, cytokines, etc.)
2. Common signaling pathways
3. Therapeutic target potential
2-3 sentences total."""
            
            summary_text = generate_with_llm(summary_prompt, max_tokens=150)
            if summary_text:
                summary_info = [line.strip() for line in summary_text.split('\n') if line.strip()]
            
            # Add remaining genes count (only if more than 32)
            if len(genes) > 32:
                details.append(f"... plus {len(genes) - 32} more genes")
        else:
            details = ["No genes specified for query"]
        
        result = {
            "title": f"Retrieved info for {len(genes)} genes",
            "details": details,
            "summary": summary_info if summary_info else ["Gene function analysis complete"],
            "queried_genes": genes,
            "tokens": random.randint(1500, 2500) + len(genes) * 50,
            "duration": f"{random.uniform(1.5, 3.0) + len(genes) * 0.1:.1f}s"
        }
        
        return {
            "success": True,
            "tool": self.name,
            "thought": f"Querying detailed info for {len(genes)} candidate genes from NCBI Gene DB",
            "action": f'NCBI Gene API call: genes={genes}',
            "result": result
        }


@register_tool
class CrisprDesignerTool(Tool):
    """Tool for designing CRISPR sgRNAs."""
    
    name = "crispr_designer"
    description = "Designs CRISPR sgRNAs. Use to design efficient guide RNAs for target genes."
    parameters = [
        ToolParameter(
            name="target_genes",
            type="array",
            description="List of target genes for sgRNA design",
            required=True
        ),
        ToolParameter(
            name="sgrnas_per_gene",
            type="number",
            description="Number of sgRNAs to design per gene",
            required=False,
            default=3
        )
    ]
    
    def execute(self, target_genes: List[str], sgrnas_per_gene: int = 3, **kwargs) -> Dict[str, Any]:
        """Execute CRISPR sgRNA design with dynamic statistics and LLM analysis."""
        total_sgrnas = len(target_genes) * sgrnas_per_gene
        
        # Generate realistic efficiency scores
        avg_efficiency = round(random.uniform(0.65, 0.85), 2)
        gc_low = random.randint(35, 45)
        gc_high = random.randint(55, 65)
        offtarget_max = random.randint(1, 4)
        
        # Generate LLM analysis for sgRNA design
        key_genes_str = ', '.join(target_genes[:5])
        analysis_prompt = f"""Write CRISPR sgRNA design analysis results:
Target genes: {key_genes_str} (total {len(target_genes)})
Designed sgRNAs: {total_sgrnas}

Provide 1-2 sentences for each:
1. Key target genes with expected high efficiency
2. Off-target risk analysis
3. Screening strategy recommendations"""
        
        llm_analysis = generate_with_llm(analysis_prompt, max_tokens=200)
        
        # Build detailed results
        details = [
            f"Target genes: {', '.join(target_genes[:6])}{'...' if len(target_genes) > 6 else ''} (total {len(target_genes)})",
            f"sgRNAs per gene: {sgrnas_per_gene} -> {total_sgrnas} total designed",
            f"Average efficiency score: {avg_efficiency} (CRISPRscan algorithm)",
            f"Off-target prediction: all below {offtarget_max} (Cas-OFFinder analysis)",
            f"GC content range: {gc_low}-{gc_high}% (within optimal range)"
        ]
        
        # Add LLM analysis
        if llm_analysis:
            for line in llm_analysis.split('\n'):
                line = line.strip()
                if line and len(line) > 10:
                    details.append(line[:150])
        
        result = {
            "title": f"Designed {total_sgrnas} sgRNAs",
            "details": details,
            "target_genes": target_genes,
            "sgrnas_per_gene": sgrnas_per_gene,
            "total_sgrnas": total_sgrnas,
            "avg_efficiency": avg_efficiency,
            "tokens": random.randint(2500, 4000),
            "duration": f"{random.uniform(3.0, 6.0):.1f}s",
            "has_graph": True,
            "graph_type": "efficiency"
        }
        
        return {
            "success": True,
            "tool": self.name,
            "thought": f"Designing {sgrnas_per_gene} efficient sgRNAs for each of {len(target_genes)} genes",
            "action": f'Running CRISPRscan algorithm: {len(target_genes)} genes × {sgrnas_per_gene} sgRNAs = {total_sgrnas} sgRNAs',
            "result": result
        }


@register_tool
class ProtocolBuilderTool(Tool):
    """Tool for building experimental protocols."""
    
    name = "protocol_builder"
    description = "Generates experimental protocols. Use to construct procedures for CRISPR screens, cell culture, etc."
    parameters = [
        ToolParameter(
            name="experiment_type",
            type="string",
            description="Experiment type (e.g., 'crispr_screen', 'cell_culture')",
            required=True,
            enum=["crispr_screen", "cell_culture", "flow_cytometry", "sequencing"]
        ),
        ToolParameter(
            name="duration_weeks",
            type="number",
            description="Experiment duration (weeks)",
            required=False,
            default=6
        )
    ]
    
    def execute(self, experiment_type: str, duration_weeks: int = 6, **kwargs) -> Dict[str, Any]:
        """Execute protocol building with experiment-specific details and LLM enhancement."""
        experiment_names = {
            "crispr_screen": "CRISPR Screen",
            "cell_culture": "Cell Culture",
            "flow_cytometry": "Flow Cytometry",
            "sequencing": "Sequencing"
        }
        exp_name = experiment_names.get(experiment_type, experiment_type)
        
        # Define protocol templates based on experiment type
        protocol_templates = {
            "crispr_screen": [
                f"Week 1-{duration_weeks//3}: Library cloning and lentivirus production",
                f"Week {duration_weeks//3+1}-{2*duration_weeks//3}: T cell transduction and selection",
                f"Week {2*duration_weeks//3+1}-{duration_weeks}: Stimulation analysis and sequencing"
            ],
            "cell_culture": [
                f"Week 1-{duration_weeks//2}: Cell culture and passage",
                f"Week {duration_weeks//2+1}-{duration_weeks}: Treatment and analysis"
            ],
            "flow_cytometry": [
                f"Week 1-{duration_weeks//3}: Sample preparation and antibody optimization",
                f"Week {duration_weeks//3+1}-{2*duration_weeks//3}: Data collection",
                f"Week {2*duration_weeks//3+1}-{duration_weeks}: Data analysis"
            ],
            "sequencing": [
                f"Week 1-{duration_weeks//4}: Library preparation",
                f"Week {duration_weeks//4+1}-{duration_weeks//2}: Sequencing run",
                f"Week {duration_weeks//2+1}-{duration_weeks}: Data analysis and interpretation"
            ]
        }
        
        base_details = protocol_templates.get(experiment_type, protocol_templates["crispr_screen"])
        
        # Generate LLM-enhanced protocol details
        protocol_prompt = f"""Write {exp_name} experimental protocol ({duration_weeks} weeks):
1. Key considerations for each step (1 sentence each)
2. Required equipment/reagents summary
3. Quality control checkpoints
4. Expected challenges and solutions
Keep it concise."""
        
        llm_details = generate_with_llm(protocol_prompt, max_tokens=250)
        
        # Build comprehensive details
        details = base_details.copy()
        
        # Add LLM-generated content
        if llm_details:
            details.append("─" * 20)  # Separator
            for line in llm_details.split('\n'):
                line = line.strip()
                if line and len(line) > 5:
                    details.append(line[:150])
        
        # Add resources and cost
        cost = random.randint(5, 20) * 1000
        details.extend([
            "─" * 20,
            f"Required personnel: {random.randint(1, 3)} researcher(s)",
            f"Estimated cost: ${cost:,}",
            f"Predicted success rate: {random.randint(75, 95)}%"
        ])
        
        result = {
            "title": f"Generated {duration_weeks}-week {exp_name} protocol",
            "details": details,
            "experiment_type": experiment_type,
            "duration_weeks": duration_weeks,
            "estimated_cost": cost,
            "tokens": random.randint(2000, 3500),
            "duration": f"{random.uniform(2.0, 4.0):.1f}s",
            "has_graph": True,
            "graph_type": "timeline"
        }
        
        return {
            "success": True,
            "tool": self.name,
            "thought": f"Generating complete {duration_weeks}-week experimental protocol for {exp_name}",
            "action": f'Generating {duration_weeks}-week {exp_name} schedule and detailed instructions based on protocol template',
            "result": result
        }
