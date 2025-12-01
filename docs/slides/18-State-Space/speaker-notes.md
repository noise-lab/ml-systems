# NetSSM DARPA Presentation - Speaker Notes

## Slide 1: Title
- Introduce the research team, emphasize University of Chicago
- This is about generating synthetic network traffic data

## Slide 2: The Challenge
- Start with the problem: we need network data for research
- But getting real data is hard: privacy laws, GDPR, collection costs
- This motivates synthetic data generation

## Slide 3: Limitations of Current Approaches
- Two types of generators exist today
- NetShare does attributes only - can't capture TCP state
- NetDiffusion does raw packets but very short traces
- **Key point**: Neither can do multi-flow sessions - this is a critical gap

## Slide 4: NetSSM Overview
- We use Mamba state-space models (newer than transformers)
- Combines best of both worlds: length + detail
- Sequential nature is perfect for network traffic

## Slide 5: Innovation 1 - Multi-Flow
- **This is a first** - emphasize novelty
- Real world is multi-flow (IoT, distributed systems, streaming)
- Give example: Netflix has CDN setup flows + video download flows happening together
- Previous work couldn't do this reliably

## Slide 6: Innovation 2 - Length Scaling
- **Numbers matter here**: 8× and 78× improvements
- 100K tokens = about 943 packets of context
- Why it matters: TCP handshake at start affects behavior later
- Can model entire sessions, not just connection setup

## Slide 7: Architecture (LARGE FIGURE)
- Walk through the pipeline left to right
- Pre-processing: PCAP → tokens
- Training: learns patterns
- Generation: produces synthetic PCAPs
- This is all automatic once trained

## Slide 8: Method Details
- Tokenization is simple: one byte = one token
- Special tokens mark packet boundaries and labels
- Training is unsupervised - just learns to predict next byte
- Batch size of 1 allows maximum sequence length

## Slide 9: Evaluation Framework
- **Three dimensions** - this is comprehensive
- Statistical similarity: traditional metric
- ML utility: does it help train models?
- **Semantic similarity: our new contribution** - does it behave correctly?

## Slide 10: Statistical Similarity Results
- Tested on 10 apps: streaming, conferencing, social media
- JSD of 0.02 is excellent (lower is better)
- **8× better than NetShare, 2× better than NetDiffusion**
- This validates basic fidelity

## Slide 11: ML Performance (LARGE FIGURE)
- This shows practical utility
- Random forest trained ONLY on synthetic data
- **97% accuracy on real test data** - this is remarkable
- Compare to 16% and 13% for competitors
- Means our synthetic data is truly useful

## Slide 12: Detailed Comparison (LARGE FIGURE)
- Shows performance across multiple metrics
- Consistently outperforms on all dimensions
- Different colors show different methods
- NetSSM (ours) wins across the board

## Slide 13: Statistical Distributions
- Shows we capture the right packet size distributions
- KDE plot matches real data closely
- This kind of fidelity is important for realism

## Slide 14: TCP Compliance
- **Semantic similarity** - new evaluation approach
- Generated traffic follows TCP rules
- Gets handshakes right, sequence numbers progress correctly
- Even captures real-world quirks (partial teardowns)
- Passes protocol validators

## Slide 15: Application Patterns
- Can handle complex multi-step protocols
- Video streaming example: CDN setup then segment downloads
- Captures application-specific timing and patterns
- This is where multi-flow capability shines

## Slide 16: Impact and Applications
- Security: train IDS without exposing real data
- Performance testing: scale testing without production traffic
- Protocol development: validate before deployment
- **Key benefit**: eliminates privacy/governance barriers

## Slide 17: Comparison Table
- Direct comparison shows NetSSM advantages
- **Only method with multi-flow capability**
- Context and generation length are crucial
- This is a significant step forward

## Slide 18: Key Takeaways
- Three main contributions to remember:
  1. Multi-flow (first of its kind)
  2. Superior performance (quantified)
  3. Semantic similarity (new evaluation)
- Synthetic data that's both similar AND useful

## Slide 19: Future Directions
- UDP and other protocols
- Encrypted payload patterns
- Adaptive generation for conditions
- Integration with simulation tools
- Standardizing semantic metrics
- Building benchmark datasets

## Slide 20: Summary
- Reinforce the three innovations
- Emphasize performance results
- Open for questions

## Slide 21: Contact
- QR code goes to website
- Available for follow-up

---

## Timing Guide (20 minutes total)
- Slides 1-4: 3 minutes (setup and problem)
- Slides 5-8: 5 minutes (approach and method)
- Slides 9-15: 8 minutes (results - this is the meat)
- Slides 16-20: 3 minutes (impact and wrap-up)
- Slide 21: 1 minute (questions)

## Key Messages to Emphasize
1. **First multi-flow generator** - this is novel
2. **97% accuracy** - this is the money shot
3. **8× and 78× improvements** - quantify the advance
4. **Semantic similarity** - new way to evaluate

## Potential Questions & Answers
**Q: How does this compare to just using real data?**
A: Real data often unavailable due to privacy. Our synthetic data achieves 97% accuracy showing it's a viable substitute.

**Q: What about UDP traffic?**
A: Currently TCP-focused, but architecture supports UDP - that's future work.

**Q: Can this be used for adversarial purposes?**
A: Like any tool, but primary use is defensive - training IDS, testing systems.

**Q: How long does training take?**
A: 30 epochs on A40 GPU. Generation is fast once trained.

**Q: Is the code available?**
A: [Defer to team decision on open source]

## DARPA-Specific Angles
- Emphasize security applications (IDS training)
- Privacy-preserving research enablement
- Scalability for testing DoD systems
- Protocol validation for new standards
- Defensive security applications
