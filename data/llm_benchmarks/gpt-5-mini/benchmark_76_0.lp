d0(o0).
d1(o1).
d5(o1).
d2(o1).
d3(o5).
d4(o3).

d4(X) :- d1(X), not d3(X).
d1(X) :- d5(X).
d5(X) :- d3(X).
d0(X) :- d3(X).
d3(X) :- d0(X).