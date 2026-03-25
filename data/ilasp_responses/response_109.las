%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.155s
%% Conflict analysis                       : 0.357s
%%   - Positive Examples                   : 0.357s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.094s
%% Total                                   : 0.635s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [62, 68, 555] 7 
%% d2(V1) :- d3(V1). 
%% d3(V1) :- d4(V1). 
%% d5(V1) :- d2(V1); not d4(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d2(V1) :- d3(V1).
d3(V1) :- d4(V1).
d5(V1) :- d2(V1); not d4(V1).

