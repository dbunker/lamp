d5(o3).
d2(o3).
d5(o5).
d4(o3).
d5(o4).

d0(X) :- d4(X), d3(X).
d0(X) :- d1(X), d5(X).
d1(X) :- d2(X), d5(X).
d5(X) :- d2(X), d4(X).
d4(X) :- d1(X), d0(X).
d3(X) :- d2(X), d0(X).
d0(X) :- d2(X), d3(X).
d3(X) :- d0(X), d5(X).
d1(X) :- d2(X), d5(X).
d1(X) :- d0(X), d5(X).
d4(X) :- d0(X), d3(X).