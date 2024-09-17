#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

namespace pico_gpt2
{
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    struct Block 
    {
        // Layer Norm 1 parameters
        Vector ln1_g;
        Vector ln1_b;
        
        // Attention parameters
        Matrix attn_w;      // [n_embd, 3 * n_embd]
        Vector attn_b;      // [3 * n_embd]
        Matrix attn_proj_w; // [n_embd, n_embd]
        Vector attn_proj_b; // [n_embd]
        
        // Layer Norm 2 parameters
        Vector ln2_g;
        Vector ln2_b;
        
        // Feed-Forward Network parameters
        Matrix ffn_fc_w;    // [n_embd, 4 * n_embd]
        Vector ffn_fc_b;    // [4 * n_embd]
        Matrix ffn_proj_w;  // [4 * n_embd, n_embd]
        Vector ffn_proj_b;  // [n_embd]
    };

    //
    // GELU
    // https://arxiv.org/pdf/1606.08415
    //
    inline Matrix 
    gelu(const Matrix &x) 
    {
        return 0.5f * x.array() * (1.0f + ((2.0f / M_PI) * (x.array() + 0.044715f * x.array().cube())).tanh());
    }

    //
    // Softmax
    // 
    //
    inline Matrix 
    softmax(const Matrix &x)
    {
        // Subtract the max for numerical stability
        Matrix shifted = x.rowwise() - x.colwise().maxCoeff();
        Matrix exp_x   = shifted.array().exp();
        Vector sum_exp = exp_x.rowwise().sum();

        return exp_x.array().colwise() / sum_exp.array();
    }

    //
    // Layer Normalization
    //
    inline Matrix 
    layer_norm(const Matrix &x, const Vector &g, const Vector &b, double eps = 1e-5) 
    {
        // Compute mean
        Vector mean = x.rowwise().mean();
        
        // Compute variance
        Matrix diff     = x.rowwise() - mean.transpose();
        Matrix sq_diff  = diff.array().square();
        Vector variance = sq_diff.rowwise().mean();
        
        // Normalize
        Matrix normalized = (diff.array().colwise() / (variance.array() + eps).sqrt().matrix().transpose().replicate(1, x.cols()).array());
        
        // Scale and shift
        Matrix scaled  = normalized.array().rowwise() * g.transpose().array();
        Matrix shifted = scaled.array().rowwise() + b.transpose().array();
        
        return shifted;
    }

    //
    // Linear (Matrix multiplication + bias)
    //
    inline Matrix 
    linear(const Matrix &x, const Matrix &w, const Vector &b) 
    {
        return (x * w).rowwise() + b.transpose();
    }

    //
    // FFN
    //
    inline Matrix 
    ffn(const Matrix &x, const Matrix &c_fc_w, const Vector &c_fc_b, const Matrix &c_proj_w, const Vector &c_proj_b) 
    {
        // Project up
        Matrix a = gelu(linear(x, c_fc_w, c_fc_b));
        
        // Project back down
        Matrix out = linear(a, c_proj_w, c_proj_b);
        
        return out;
    }

    //
    // Attention
    //
    inline Matrix 
    attention(const Matrix &q, const Matrix &k, const Matrix &v, const Matrix &mask) 
    {
        // Compute scaled dot-product
        Matrix scores = (q * k.transpose()) / std::sqrt(static_cast<double>(q.cols()));
        
        // Apply mask
        Matrix masked_scores = scores.array() + mask.array();
        
        // Softmax
        Matrix attn_weights = softmax(masked_scores);
        
        // Weighted sum
        return attn_weights * v;
    }
    
    inline Matrix 
    mha(const Matrix &x, const Matrix &c_attn_w, const Vector &c_attn_b, const Matrix &c_proj_w, const Vector &c_proj_b, int n_head) 
    {
        // Linear projection to get Q, K, V
        Matrix qkv = linear(x, c_attn_w, c_attn_b); // [n_seq, 3 * n_embd]
        
        int n_embd = qkv.cols() / 3;
        Matrix q = qkv.leftCols(n_embd);
        Matrix k = qkv.middleCols(n_embd, n_embd);
        Matrix v = qkv.rightCols(n_embd);
        
        // Split into heads
        int head_dim = n_embd / n_head;
        auto split_heads = [&](const Matrix& mat) -> std::vector<Matrix>
        {
            std::vector<Matrix> heads;
            for(int i = 0; i < n_head; ++i) 
            {
                heads.emplace_back(mat.block(0, i * head_dim, mat.rows(), head_dim));
            }

            return heads;
        };
        
        std::vector<Matrix> q_heads = split_heads(q);
        std::vector<Matrix> k_heads = split_heads(k);
        std::vector<Matrix> v_heads = split_heads(v);
        
        // Create causal mask
        int n_seq = x.rows();
        Matrix causal_mask = Matrix::Zero(n_seq, n_seq);

        for (int i = 0; i < n_seq; ++i) 
        {
            for(int j = 0; j < n_seq; ++j) 
            {
                if(j > i) 
                {
                    causal_mask(i, j) = -1e10;
                }
            }
        }
        
        // Perform attention for each head
        std::vector<Matrix> out_heads;
        for (int i = 0; i < n_head; ++i) 
        {
            Matrix out = attention(q_heads[i], k_heads[i], v_heads[i], causal_mask);
            out_heads.emplace_back(out);
        }
        
        // Concatenate heads
        Matrix concatenated = Matrix::Zero(n_seq, n_embd);
        for (int i = 0; i < n_head; ++i) 
        {
            concatenated.block(0, i * head_dim, n_seq, head_dim) = out_heads[i];
        }
        
        // Final linear projection
        Matrix proj = linear(concatenated, c_proj_w, c_proj_b);
        
        return proj;
    }

    //
    // Transformer Block
    //
    inline Matrix 
    transformer_block(const Matrix &x, const Block &block, int n_head) 
    {
        // Layer Norm 1
        Matrix ln1 = layer_norm(x, block.ln1_g, block.ln1_b);
        
        // Multi-Head Attention
        Matrix mha_out = mha(ln1, block.attn_w, block.attn_b, block.attn_proj_w, block.attn_proj_b, n_head);
        
        // Residual Connection
        Matrix x1 = x + mha_out;
        
        // Layer Norm 2
        Matrix ln2 = layer_norm(x1, block.ln2_g, block.ln2_b);
        
        // Feed-Forward Network
        Matrix ffn_out = ffn(ln2, block.ffn_fc_w, block.ffn_fc_b, block.ffn_proj_w, block.ffn_proj_b);
        
        // Residual Connection
        Matrix x2 = x1 + ffn_out;
        
        return x2;
    }

    inline Matrix 
    gpt2(const std::vector<int> &inputs, 
         const Matrix &wte, const Matrix &wpe, 
         const std::vector<Block> &blocks, 
         const Vector &ln_f_g, const Vector &ln_f_b, 
         const Matrix &final_w, 
         int n_head) 
    {
        int n_seq  = inputs.size();
        int n_embd = wte.cols();
        
        // Token Embeddings
        Matrix token_emb(n_seq, n_embd);
        for(int i = 0; i < n_seq; ++i) 
        {
            token_emb.row(i) = wte.row(inputs[i]);
        }
        
        // Positional Embeddings
        Matrix pos_emb(n_seq, n_embd);
        for(int i = 0; i < n_seq; ++i) 
        {
            pos_emb.row(i) = wpe.row(i);
        }
        
        // Combine token and positional embeddings
        Matrix x = token_emb + pos_emb;
        
        // Forward pass through transformer blocks
        for(const auto& block : blocks) 
        {
            x = transformer_block(x, block, n_head);
        }
        
        // Final Layer Norm
        Matrix ln_f = layer_norm(x, ln_f_g, ln_f_b);
        
        // Projection to vocabulary
        Matrix logits = ln_f * final_w.transpose();
        
        return logits;
    }

    inline std::vector<int> 
    generate(std::vector<int> inputs, 
             const Matrix &wte, const Matrix &wpe, 
             const std::vector<Block> &blocks, 
             const Vector &ln_f_g, const Vector &ln_f_b, 
             const Matrix &final_w, 
             int n_head, 
             int n_tokens_to_generate) 
    {
        for (int i = 0; i < n_tokens_to_generate; ++i) 
        {
            // Forward pass
            Matrix logits = gpt2(inputs, wte, wpe, blocks, ln_f_g, ln_f_b, final_w, n_head);
            
            // Get the last token's logits
            Vector last_logits = logits.row(logits.rows() - 1);
            
            // Greedy sampling
            int next_id;
            last_logits.maxCoeff(&next_id);
            
            // Append to inputs
            inputs.push_back(next_id);
        }

        // Return the generated tokens
        std::vector<int> generated(inputs.end() - n_tokens_to_generate, inputs.end());
        return generated;
    }

}
