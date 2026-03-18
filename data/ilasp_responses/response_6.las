%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.341s
%% Conflict analysis                       : 49.043s
%%   - Positive Examples                   : 49.043s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 16.87s
%% Total                                   : 69.545s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d7(V1) :- d1(V1).
d6(V1) :- d7(V1).
d5(V1) :- d1(V1).
d4(V1) :- d8(V1).
d2(V1) :- d3(V1).
d3(V1) :- d4(V1).

