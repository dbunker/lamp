%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 2.035s
%% Conflict analysis                       : 257.071s
%%   - Positive Examples                   : 257.071s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 24.715s
%% Total                                   : 284.989s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [20, 168, 210, 2552, 9241, 11665] 17 
%% d5(V1) :- d2(V1). 
%% d1(V1) :- d3(V1). 
%% d9(V1) :- d5(V1). 
%% d0(V1) :- d1(V1); not d2(V1). 
%% d1(V1) :- d9(V1); not d4(V1); not d7(V1). 
%% d6(V1) :- d9(V1); not d5(V1); not d7(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d5(V1) :- d2(V1).
d1(V1) :- d3(V1).
d9(V1) :- d5(V1).
d0(V1) :- d1(V1); not d2(V1).
d1(V1) :- d9(V1); not d4(V1); not d7(V1).
d6(V1) :- d9(V1); not d5(V1); not d7(V1).

