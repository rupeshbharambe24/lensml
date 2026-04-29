"""Tool registry — 12 tools, all read-only except `propose_fix`.

Each tool returns ≤2KB summary stats; never raw tensors. (Plan §5.2.)

    inspect_loss_curve(window, smooth)
    inspect_gradient_flow(layer_pattern, last_n_steps)
    inspect_activations(layer, sample_size)
    inspect_data_batch(n)
    inspect_dataloader_throughput()
    inspect_optimizer_state()
    inspect_lr_schedule()
    inspect_model_summary()
    inspect_train_val_gap(metric)
    inspect_memory()
    read_training_script(start_line, end_line)
    propose_fix(template, params, confidence, rationale)  # write-side only

`propose_fix` emits an ApprovalRequest; it never applies a patch directly.
The git + AST + restart sequence is owned by the server, not the agent.
"""

from __future__ import annotations

# Implemented Day 9.
