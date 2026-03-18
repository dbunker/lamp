%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.423s
%% Conflict analysis                       : 2.436s
%%   - Positive Examples                   : 2.436s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.764s
%% Total                                   : 3.73s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d6(V1) :- d4(V1).
d8(V1) :- d5(V1).
d9(V1) :- d8(V1).
d7(V1) :- d8(V1).
d1(V1) :- d9(V1); not d8(V1).

