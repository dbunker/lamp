d2(o4).
d1(o0).
d2(o0).
d1(o2).
d3(o6).

d2(X) :- not d0(X), d5(X), d3(X).
d0(X) :- d2(X), not d5(X), d4(X).
d4(X) :- d1(X), not d3(X), d5(X).
d3(X) :- d2(X), d0(X), not d5(X).
d1(X) :- d2(X), d0(X), not d4(X).
d2(X) :- d4(X), d1(X), not d0(X).
d5(X) :- not d0(X), d1(X), d3(X).
d3(X) :- d2(X), d1(X), not d0(X).
d3(X) :- d1(X), d5(X), not d2(X).
d3(X) :- d2(X), d1(X), not d4(X).
d0(X) :- d4(X), not d5(X), d3(X).