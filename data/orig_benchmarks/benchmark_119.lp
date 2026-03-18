d3(o1).
d2(o3).
d4(o5).
d3(o3).
d3(o0).

d2(X) :- d4(X), d1(X).
d3(X) :- d1(X), d0(X).
d1(X) :- d4(X), d0(X).
d0(X) :- d2(X), d3(X).
d1(X) :- d0(X), d3(X).
d0(X) :- d4(X), d5(X).
d5(X) :- d4(X), d3(X).
d0(X) :- d2(X), d4(X).
d3(X) :- d2(X), d4(X).
d0(X) :- d2(X), d3(X).
d1(X) :- d4(X), d0(X).