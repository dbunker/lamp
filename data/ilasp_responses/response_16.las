%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.481s
%% Conflict analysis                       : 17.529s
%%   - Positive Examples                   : 17.529s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 6.775s
%% Total                                   : 26.662s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [50, 55, 67, 105, 111, 151, 2097] 15 
%% d4(V1) :- d3(V1). 
%% d7(V1) :- d3(V1). 
%% d1(V1) :- d3(V1). 
%% d5(V1) :- d0(V1). 
%% d4(V1) :- d0(V1). 
%% d8(V1) :- d6(V1). 
%% d2(V1) :- d4(V1); not d0(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d4(V1) :- d3(V1).
d7(V1) :- d3(V1).
d1(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d8(V1) :- d6(V1).
d2(V1) :- d4(V1); not d0(V1).

