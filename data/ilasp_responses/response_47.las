%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.095s
%% Conflict analysis                       : 0.246s
%%   - Positive Examples                   : 0.246s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.085s
%% Total                                   : 0.447s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [12, 23, 27, 54, 217] 11 
%% d3(V1) :- d1(V1). 
%% d4(V1) :- d3(V1). 
%% d5(V1) :- d0(V1). 
%% d4(V1) :- d0(V1). 
%% d0(V1) :- d4(V1); not d3(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d3(V1) :- d1(V1).
d4(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d0(V1) :- d4(V1); not d3(V1).

