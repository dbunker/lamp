d4(o1).
d5(o1).
d1(o3).
d0(o5).
d2(o4).
d4(o5).

d0(X) :- d2(X), d3(X), not d1(X).
d2(X) :- d4(X), d1(X), not d5(X).
d1(X) :- d2(X), d0(X), not d3(X).
d1(X) :- d4(X), not d5(X), d3(X).
d5(X) :- d2(X), d1(X), not d3(X).
d4(X) :- d2(X), d1(X), not d3(X).
d0(X) :- d4(X), d3(X), not d2(X).
d3(X) :- d4(X), d0(X), not d5(X).
d4(X) :- d5(X), d3(X), not d2(X).
d0(X) :- d2(X), not d3(X), d5(X).
d2(X) :- d1(X), d0(X), not d4(X).