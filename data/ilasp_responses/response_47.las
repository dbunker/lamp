%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.095s
%% Conflict analysis                       : 0.258s
%%   - Positive Examples                   : 0.258s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.084s
%% Total                                   : 0.458s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d1(V1).
d4(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d0(V1) :- d4(V1); not d3(V1).

