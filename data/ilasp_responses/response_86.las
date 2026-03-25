%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.15s
%% Conflict analysis                       : 0.388s
%%   - Positive Examples                   : 0.388s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.138s
%% Total                                   : 0.707s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [22, 46, 69, 531] 9 
%% d5(V1) :- d2(V1). 
%% d5(V1) :- d0(V1). 
%% d4(V1) :- d0(V1). 
%% d3(V1) :- d4(V1); not d0(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d5(V1) :- d2(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d3(V1) :- d4(V1); not d0(V1).

