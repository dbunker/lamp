%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.196s
%% Conflict analysis                       : 0.663s
%%   - Positive Examples                   : 0.663s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.195s
%% Total                                   : 1.094s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d3(V1).
d5(V1) :- d4(V1).
d1(V1) :- d0(V1).
d4(V1) :- d2(V1).

