# Under-Construction
The 2018 World Model that I implemented last time failed to achieve satisfactory results due to prediction errors in world model, long training time of LSTM, inefficiency of gradient-free ES, and other practical limitations. I found Dreamer V1–V3, improved versions of the 2018 World Model, which demonstrated their ability to perform articulation control, so I decided to implement Dreamer rather than continuing with the previous one.

Although I continued trying to implement V1, it failed to achieve any meaningful performance, and I could not find out what caused the problem. So, I tried another approach that was using SAC that I had previously implemented as a starting point, and I could find out what caused the problem: RSSM (Recurrent State-Space Model).

