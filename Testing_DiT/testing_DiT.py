
# =======================================
# TESTING CODE
# =======================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DiT HOOK TESTING")
    print("="*80 + "\n")
    
    # Create a minimal DiT model for testing
    print("Creating DiT model...")
    model = DiT(
        dim=128,
        depth=4,
        heads=4,
        dim_head=32,
        mel_dim=100,
        text_num_embeds=256,
        dropout=0.0
    )
    print("✓ Model created\n")
    
    # Test 1: Print available submodules
    print("\n" + "─"*80)
    print("TEST 1: Available Submodules")
    print("─"*80)
    model.print_available_submodules(max_depth=2)
    
    # Test 2: Create dummy data
    print("\n" + "─"*80)
    print("TEST 2: Running Forward Pass")
    print("─"*80)
    batch_size, seq_len, mel_dim = 2, 50, 100
    x = torch.randn(batch_size, seq_len, mel_dim)
    cond = torch.randn(batch_size, seq_len, mel_dim)
    text = torch.randint(0, 256, (batch_size, seq_len))
    time = torch.tensor(0.5)
    
    print(f"Input shapes:")
    print(f"  x:    {x.shape}")
    print(f"  cond: {cond.shape}")
    print(f"  text: {text.shape}")
    print(f"  time: {time.shape}\n")
    
    # Test 3: Add extraction hook to various layers
    print("\n" + "─"*80)
    print("TEST 3: Extraction Hook on Multiple Layers")
    print("─"*80)
    model.add_extraction_hook('transformer_blocks.0', 'block0_output')
    model.add_extraction_hook('transformer_blocks.0.attn', 'block0_attn_output')
    model.add_extraction_hook('transformer_blocks.0.ff', 'block0_ff_output')
    model.add_extraction_hook('input_embed', 'input_embed_output')
    model.add_extraction_hook('input_embed.proj', 'input_proj_output')
    
    output = model(x, cond, text, time)
    print(f"\nOutput shape: {output.shape}")
    print(model)
    
    stats = model.get_hook_stats()
    print("Hook Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    # Test 4: Patch hook on projection layer (simpler target)
    print("\n" + "─"*80)
    print("TEST 4: Patch Hook - Zero First Batch on Input Projection")
    print("─"*80)
    model.remove_hooks()
    
    # Add BOTH extraction and patch hooks to verify
    model.add_extraction_hook('input_embed.proj', 'before_patch')
    output_before = model(x, cond, text, time)
    before_data = model.extracted_activations['before_patch']['tensor']
    print(f"Before patch - Proj mean: {before_data.mean():.6f}, std: {before_data.std():.6f}")
    print(f"Before patch - Batch 0 mean: {before_data[0].mean():.6f}")
    print(f"Before patch - Batch 1 mean: {before_data[1].mean():.6f}")
    
    model.remove_hooks()
    model.add_patch_hook('input_embed.proj', 'zero_batch0', 
                         batch_index=0, patch_vec=0.0)
    model.add_extraction_hook('input_embed.proj', 'after_patch')
    
    output_patched = model(x, cond, text, time)
    after_data = model.extracted_activations['after_patch']['tensor']
    print(f"\nAfter patch  - Proj mean: {after_data.mean():.6f}, std: {after_data.std():.6f}")
    print(f"After patch  - Batch 0 mean: {after_data[0].mean():.6f}, max: {after_data[0].abs().max():.6f}")
    print(f"After patch  - Batch 1 mean: {after_data[1].mean():.6f}, std: {after_data[1].std():.6f}")
    print(f"\nOutput difference from original: {(output_before - output_patched).abs().mean():.6f}")
    print(f"Output difference max: {(output_before - output_patched).abs().max():.6f}")
    print(f"Patch hook was called {model.hook_objects['zero_batch0'].call_count} time(s)")
    
    if after_data[0].abs().max() < 1e-6:
        print("✓ PATCH WORKING: Batch 0 successfully zeroed!")
    else:
        print("✗ PATCH FAILED: Batch 0 was not zeroed!")
    
    # Test 5: Patch hook - zero specific feature dimension (with verification)
    print("\n" + "─"*80)
    print("TEST 5: Patch Hook - Zero Feature Index 10")
    print("─"*80)
    model.remove_hooks()
    
    # Before patch
    model.add_extraction_hook('transformer_blocks.2', 'before_feat_patch')
    output_before_feat = model(x, cond, text, time)
    before_feat_data = model.extracted_activations['before_feat_patch']['tensor']
    print(f"Before patch - Feature 10 mean: {before_feat_data[:, :, 10].mean():.6f}")
    print(f"Before patch - Feature 10 std:  {before_feat_data[:, :, 10].std():.6f}")
    
    model.remove_hooks()
    model.add_patch_hook('transformer_blocks.2', 'zero_feature10',
                         feature_index=10, patch_vec=0.0)
    model.add_extraction_hook('transformer_blocks.2', 'after_feat_patch')
    
    output_patched2 = model(x, cond, text, time)
    after_feat_data = model.extracted_activations['after_feat_patch']['tensor']
    print(f"After patch  - Feature 10 mean: {after_feat_data[:, :, 10].mean():.6f}")
    print(f"After patch  - Feature 10 max:  {after_feat_data[:, :, 10].abs().max():.6f}")
    print(f"\nOutput difference from original: {(output_before_feat - output_patched2).abs().mean():.6f}")
    
    if after_feat_data[:, :, 10].abs().max() < 1e-6:
        print("PATCH WORKING: Feature 10 successfully zeroed!")
    else:
        print("PATCH FAILED: Feature 10 was not zeroed!")
    
    # Test 6: Test patch on input_embed (simpler layer)
    print("\n" + "─"*80)
    print("TEST 6: Patch Hook on Input Embedding (Direct Test)")
    print("─"*80)
    model.remove_hooks()
    
    # Capture input_embed output before patch
    model.add_extraction_hook('input_embed', 'input_before')
    output_clean = model(x, cond, text, time)
    input_before = model.extracted_activations['input_before']['tensor']
    print(f"Input embed before - mean: {input_before.mean():.6f}")
    print(f"Input embed before - Batch 0 mean: {input_before[0].mean():.6f}")
    
    # Now patch it
    model.remove_hooks()
    model.add_patch_hook('input_embed', 'patch_input_batch0', 
                         batch_index=0, patch_vec=0.0)
    model.add_extraction_hook('input_embed', 'input_after')
    
    output_patched_input = model(x, cond, text, time)
    input_after = model.extracted_activations['input_after']['tensor']
    print(f"\nInput embed after  - mean: {input_after.mean():.6f}")
    print(f"Input embed after  - Batch 0 mean: {input_after[0].mean():.6f}")
    print(f"Input embed after  - Batch 0 max:  {input_after[0].abs().max():.6f}")
    print(f"Input embed after  - Batch 1 mean: {input_after[1].mean():.6f}")
    
    print(f"\nOutput difference: {(output_clean - output_patched_input).abs().mean():.6f}")
    
    if input_after[0].abs().max() < 1e-6:
        print("PATCH WORKING: Input batch 0 successfully zeroed!")
    else:
        print("PATCH FAILED: Input batch 0 was not zeroed!")
    
    # Test 7: Multiple hooks
    print("\n" + "─"*80)
    print("TEST 7: Multiple Hooks Simultaneously")
    print("─"*80)
    model.remove_hooks()
    model.add_extraction_hook('transformer_blocks.0', 'block0')
    model.add_extraction_hook('transformer_blocks.2', 'block2')
    model.add_patch_hook('transformer_blocks.1', 'zero_b0_f5',
                         batch_index=0, feature_index=5, patch_vec=0.0)
    
    output_multi = model(x, cond, text, time)
    print(model)
    
    final_stats = model.get_hook_stats()
    print("\nFinal Hook Statistics:")
    for key, val in final_stats.items():
        print(f"  {key}: {val}")
    
    # Clean up
    model.remove_hooks()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
