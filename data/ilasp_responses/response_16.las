%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.473s
%% Conflict analysis                       : 17.432s
%%   - Positive Examples                   : 17.432s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 6.179s
%% Total                                   : 25.881s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d3(V1).
d7(V1) :- d3(V1).
d1(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d8(V1) :- d6(V1).
d2(V1) :- d4(V1); not d0(V1).

