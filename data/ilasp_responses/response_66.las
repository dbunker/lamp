%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.076s
%% Conflict analysis                       : 0.161s
%%   - Positive Examples                   : 0.161s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.055s
%% Total                                   : 0.31s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [40, 49, 52, 326] 9 
%% d2(V1) :- d4(V1). 
%% d2(V1) :- d5(V1). 
%% d0(V1) :- d5(V1). 
%% d4(V1) :- d0(V1); not d1(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d2(V1) :- d4(V1).
d2(V1) :- d5(V1).
d0(V1) :- d5(V1).
d4(V1) :- d0(V1); not d1(V1).

